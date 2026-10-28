import argparse
import torch
from model import Config, Transformer

p = argparse.ArgumentParser(description="Generate KJV-style text, not verified scripture")
p.add_argument("--checkpoint", default="runs/kjv/best.pt")
p.add_argument("--prompt", default="In the beginning")
p.add_argument("--bytes", type=int, default=400)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=40)
a = p.parse_args()
checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
model = Transformer(Config(**checkpoint["config"]))
model.load_state_dict(checkpoint["model"])
model.eval()
ids = torch.tensor([list((a.prompt or "\n").encode("utf-8"))], dtype=torch.long)
result = model.generate(ids, a.bytes, a.temperature, a.top_k)
print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))
