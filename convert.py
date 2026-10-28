"""Move model weights between PyTorch and the browser. Optimizer state is not translated."""
import argparse
import base64
from dataclasses import asdict
import json
from pathlib import Path
import struct
import torch
from model import Config, Transformer


def browser_key(key):
    return key.removesuffix(".weight")


def to_browser(checkpoint):
    specs, buffers = [], []
    for name, tensor in checkpoint["model"].items():
        # PyTorch linear: [out, in]; TF.js matmul kernel: [in, out].
        if tensor.ndim == 2 and name != "embed.weight":
            tensor = tensor.t()
        tensor = tensor.contiguous().float().cpu()
        values = tensor.flatten().tolist()
        buffers.append(struct.pack(f"<{len(values)}f", *values))
        specs.append({"name": browser_key(name), "shape": list(tensor.shape), "dtype": "float32"})
    args = checkpoint.get("args", {})
    step = checkpoint.get("step", 0)
    return {"format": "kjv-1m-v1", "config": checkpoint["config"], "weights": {
        "specs": specs, "data": base64.b64encode(b"".join(buffers)).decode("ascii")},
        "state": {"steps": step, "bytesSeen": step * args.get("batch_size", 1) * args.get("accum", 1) * checkpoint["config"]["context"],
                  "loss": None, "val": None, "history": [], "samplerState": 42,
                  "settings": {"batch": 1, "context": checkpoint["config"]["context"], "lr": args.get("lr", 5e-4)},
                  "manifest": checkpoint["manifest"]}}


def to_torch(packed):
    if packed["format"] != "kjv-1m-v1":
        raise ValueError("Unknown checkpoint format")
    model = Transformer(Config(**packed["config"]))
    data = bytearray(base64.b64decode(packed["weights"]["data"], validate=True))
    offset, weights = 0, {}
    for spec in packed["weights"]["specs"]:
        if spec["dtype"] != "float32":
            raise ValueError("Only float32 model tensors are supported")
        size = 1
        for n in spec["shape"]:
            size *= n
        end = offset + size * 4
        tensor = torch.frombuffer(data[offset:end], dtype=torch.float32).clone().reshape(spec["shape"])
        name = spec["name"] + ".weight"
        if tensor.ndim == 2 and name != "embed.weight":
            tensor = tensor.t().contiguous()
        if name in weights:
            raise ValueError("Duplicate tensor")
        weights[name] = tensor
        offset = end
    if offset != len(data):
        raise ValueError("Unexpected weight data length")
    model.load_state_dict(weights, strict=True)
    return {"model": model.state_dict(), "config": asdict(model.cfg),
            "manifest": packed["state"]["manifest"], "step": packed["state"]["steps"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("direction", choices=["to-browser", "to-torch"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.direction == "to-browser":
        result = to_browser(torch.load(a.input, map_location="cpu", weights_only=True))
        a.output.write_text(json.dumps(result, separators=(",", ":")))
    else:
        torch.save(to_torch(json.loads(a.input.read_text())), a.output)
    print(f"Saved {a.output}; optimizer state restarts when switching frameworks.")
