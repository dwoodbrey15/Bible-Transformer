"""Byte-level causal transformer. No pretrained weights or tokenizer."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 256
    dim: int = 64
    layers: int = 4
    heads: int = 4
    hidden: int = 160
    context: int = 256
    rope_base: float = 10000.0
    eps: float = 1e-5

    def __post_init__(self):
        assert self.dim % self.heads == 0
        assert (self.dim // self.heads) % 2 == 0
        assert self.vocab_size == 256, "This project uses a fixed UTF-8 byte tokenizer"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attn_norm = RMSNorm(cfg.dim, cfg.eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.eps)
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.gate = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, d = x.shape
        h = self.cfg.heads
        a = self.attn_norm(x)
        q, k, v = [p(a).view(b, t, h, d // h).transpose(1, 2) for p in (self.wq, self.wk, self.wv)]
        def rotate(z):
            left, right = z.chunk(2, dim=-1)
            return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
        a = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        x = x + self.wo(a.transpose(1, 2).contiguous().view(b, t, d))
        a = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(a)) * self.up(a))


class Transformer(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab_size, c.dim)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = RMSNorm(c.dim, c.eps)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.wo.weight, std=0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * c.layers))
        hd = c.dim // c.heads
        angles = torch.outer(torch.arange(c.context).float(), c.rope_base ** (-torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None], persistent=False)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, targets=None):
        t = ids.shape[1]
        if not 0 < t <= self.cfg.context:
            raise ValueError(f"Sequence length must be 1..{self.cfg.context}")
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.cos[:, :, :t], self.sin[:, :, :t])
        logits = F.linear(self.norm(x), self.embed.weight)
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids, new_bytes=300, temperature=0.8, top_k=40):
        if temperature <= 0 or not 1 <= top_k <= 256:
            raise ValueError("temperature must be positive and top_k must be 1..256")
        for _ in range(new_bytes):
            logits, _ = self(ids[:, -self.cfg.context:])
            scores = logits[:, -1] / temperature
            threshold = torch.topk(scores, top_k).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
            ids = torch.cat((ids, torch.multinomial(scores.softmax(-1), 1)), dim=1)
        return ids


if __name__ == "__main__":
    m = Transformer()
    print(asdict(m.cfg))
    print(f"{m.parameter_count():,} trainable parameters")
