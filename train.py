"""Train from random initialization on the supplied KJV byte streams."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from model import Config, Transformer


def atomic_save(payload, path):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("runs/kjv"))
    p.add_argument("--steps", type=int, default=10000, help="Total steps, including resumed steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--context", type=int, default=256, help="UTF-8 bytes, not words")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--patience", type=int, default=10, help="Stop after this many evaluations without improvement; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--resume", type=Path)
    p.add_argument("--init", type=Path, help="Initialize weights from a checkpoint, with a new optimizer/schedule")
    a = p.parse_args()
    if a.resume and a.init:
        p.error("Choose either --resume or --init")
    if min(a.steps, a.batch_size, a.accum, a.context, a.eval_every, a.eval_batches) <= 0 or a.lr <= 0:
        p.error("steps, batch size, accumulation, context, evaluation settings, and lr must be positive")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = json.loads((a.data / "manifest.json").read_text())
    streams = {}
    for name in ("train", "val"):
        raw = (a.data / f"{name}.bin").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["streams"][name]["sha256"]:
            raise ValueError(f"{name}.bin does not match manifest")
        if len(raw) <= a.context:
            raise ValueError("Corpus is shorter than context")
        streams[name] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    model = Transformer(Config(context=a.context)).to(device)
    if a.init:
        initial = torch.load(a.init, map_location="cpu", weights_only=True)
        if initial["config"] != asdict(model.cfg) or initial["manifest"] != manifest:
            raise ValueError("Initialization checkpoint model/dataset mismatch")
        model.load_state_dict(initial["model"])
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [v for v in params if v.ndim >= 2], "weight_decay": a.weight_decay},
        {"params": [v for v in params if v.ndim < 2], "weight_decay": 0.0},
    ], lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    sampler = torch.Generator().manual_seed(a.seed)
    first, best, stale = 0, float("inf"), 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=True)
        if checkpoint["config"] != asdict(model.cfg):
            raise ValueError("Model configuration differs from checkpoint")
        if checkpoint["manifest"] != manifest:
            raise ValueError("Dataset differs from checkpoint")
        for key in ("steps", "lr", "warmup", "batch_size", "accum", "weight_decay", "seed"):
            if checkpoint["args"][key] != vars(a)[key]:
                raise ValueError(f"Keep --{key.replace('_', '-')} unchanged when resuming")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first, best, stale = checkpoint["step"], checkpoint["best_val"], checkpoint["stale"]
        sampler.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(asdict(model.cfg), indent=2))
    def batch(split, generator):
        corpus = streams[split]
        starts = torch.randint(len(corpus) - a.context, (a.batch_size,), generator=generator)
        windows = torch.stack([corpus[i:i + a.context + 1] for i in starts.tolist()]).long().to(device)
        return windows[:, :-1], windows[:, 1:]
    @torch.no_grad()
    def evaluate():
        model.eval()
        # Identical validation windows every time; no effect on training's RNG.
        gen = torch.Generator().manual_seed(12345)
        losses = [model(*batch("val", gen))[1].item() for _ in range(a.eval_batches)]
        model.train()
        return sum(losses) / len(losses)
    def save(step):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=asdict(model.cfg),
                    manifest=manifest, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    step=step, best_val=best, stale=stale, sampler_rng=sampler.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None)
    print(f"{model.parameter_count():,} parameters; {device}; {len(streams['train']):,} training bytes", flush=True)
    print("Validation loss is nats/byte; bits/byte = loss / ln(2).", flush=True)
    started = time.monotonic()
    completed = first
    model.train()
    try:
        for step in range(first, a.steps):
            warm = min(1.0, (step + 1) / max(1, a.warmup))
            progress = min(1.0, max(0.0, (step - a.warmup) / max(1, a.steps - a.warmup)))
            lr = a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(a.accum):
                _, loss = model(*batch("train", sampler))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; try a smaller learning rate")
                total += loss.item() / a.accum
                (loss / a.accum).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            if completed % 10 == 0 or completed == 1:
                elapsed = time.monotonic() - started
                speed = (completed - first) * a.batch_size * a.accum * a.context / elapsed
                print(f"step {completed} train {total:.4f} lr {lr:.2g} {speed:.0f} bytes/s", flush=True)
            if completed % a.eval_every == 0 or completed == a.steps:
                val = evaluate()
                improved = val < best
                best, stale = (val, 0) if improved else (best, stale + 1)
                checkpoint = save(completed)
                atomic_save(checkpoint, a.out / "last.pt")
                if improved:
                    atomic_save(checkpoint, a.out / "best.pt")
                record = {"step": completed, "train_loss": total, "val_loss": val, "val_bits_per_byte": val / math.log(2), "lr": lr}
                with (a.out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"validation {val:.4f}; {record['val_bits_per_byte']:.3f} bits/byte; best {best:.4f}", flush=True)
                if a.patience and stale >= a.patience:
                    print("Early stopping; use best.pt for generation.")
                    break
    except KeyboardInterrupt:
        # Only save completed optimizer updates. A partial accumulation consumed RNG;
        # discard it and fall back to the last atomic evaluation checkpoint.
        print("Interrupted. Resume from the last saved evaluation checkpoint.", flush=True)


if __name__ == "__main__":
    main()
