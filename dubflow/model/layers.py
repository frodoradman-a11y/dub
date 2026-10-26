"""Transformer building blocks: RMSNorm, RoPE, attention, SwiGLU, AdaLN-zero blocks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight if self.weight is not None else x


def rope_freqs(seq_len: int, head_dim: int, device, base: float = 10000.0, positions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = positions.float() if positions is not None else torch.arange(seq_len, device=device).float()
    ang = pos[..., None] * inv  # [..., T, hd/2]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, hd]; cos/sin [T, hd/2] or [B, T, hd/2]."""
    if cos.dim() == 3:
        cos, sin = cos[:, None], sin[:, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None, attn_mask: torch.Tensor | None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = (t.float()[:, None] * 1000.0) * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], -1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class AdaLNBlock(nn.Module):
    """DiT block with AdaLN-zero modulation from a global conditioning vector."""

    def __init__(self, dim: int, heads: int, mlp_mult: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, affine=False)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim, affine=False)
        self.mlp = SwiGLU(dim, mlp_mult)
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope, attn_mask) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        x = x + g1 * self.attn(self.norm1(x) * (1 + sc1) + s1, rope, attn_mask)
        x = x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + s2)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block without global conditioning (text encoder)."""

    def __init__(self, dim: int, heads: int, mlp_mult: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult)

    def forward(self, x: torch.Tensor, rope, attn_mask) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, attn_mask)
        return x + self.mlp(self.norm2(x))


def build_attn_mask(valid: torch.Tensor, local_window: int | None = None) -> torch.Tensor:
    """valid [B, T] bool -> SDPA boolean mask [B, 1, T, T] (True = attend).

    Queries on padded positions still attend to themselves to avoid NaNs.
    """
    b, t = valid.shape
    mask = valid[:, None, None, :].expand(b, 1, t, t).clone()
    if local_window is not None and local_window > 0:
        idx = torch.arange(t, device=valid.device)
        band = (idx[None, :] - idx[:, None]).abs() <= local_window
        mask = mask & band[None, None]
    eye = torch.eye(t, dtype=torch.bool, device=valid.device)[None, None]
    return mask | eye
