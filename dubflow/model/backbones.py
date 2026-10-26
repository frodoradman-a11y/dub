"""Two backbones sharing one interface.

- DiTBackbone (DubFlow-M): AdaLN-zero DiT with U-ViT long skips, alternating
  local/global self-attention, and periodic timbre-window cross-attention.
- MultiRateUNet (DubFlow-S): Zipformer-inspired multi-rate U-Net transformer
  (25 -> 12.5 -> 6.25 -> 12.5 -> 25 Hz) with learnable bypass scales.

Interface:  forward(h [B,T,d], c [B,d], valid [B,T], timbre_fn) -> (h, repa_feats)
`timbre_fn(i, h)` returns the timbre-window cross-attention residual for layer i
at full frame rate (or None).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import AdaLNBlock, build_attn_mask, rope_freqs


def _run(blk, grad_ckpt: bool, *args):
    if grad_ckpt and torch.is_grad_enabled():
        return checkpoint(blk, *args, use_reentrant=False)
    return blk(*args)


class DiTBackbone(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_mult: float = 4.0,
                 local_window: int = 64, local_every: int = 2, timbre_every: int = 2,
                 repa_layers: tuple[int, ...] = ()):
        super().__init__()
        assert depth % 2 == 0, "U-ViT skips need an even depth"
        self.dim, self.depth, self.heads = dim, depth, heads
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads, mlp_mult) for _ in range(depth))
        self.skip_proj = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(depth // 2))
        self.local_window = local_window
        self.local_every = local_every
        self.timbre_layers = [i for i in range(depth) if timbre_every and i % timbre_every == 0]
        self.repa_layers = tuple(repa_layers)
        self.grad_ckpt = False

    def forward(self, h: torch.Tensor, c: torch.Tensor, valid: torch.Tensor,
                timbre_fn: Callable[[int, torch.Tensor], torch.Tensor | None] | None = None):
        rope = rope_freqs(h.shape[1], self.dim // self.heads, h.device)
        global_mask = build_attn_mask(valid)
        local_mask = build_attn_mask(valid, self.local_window) if self.local_every else global_mask
        skips, repa = [], {}
        half = self.depth // 2
        for i, blk in enumerate(self.blocks):
            if i >= half:
                h = self.skip_proj[i - half](torch.cat([h, skips.pop()], -1))
            if timbre_fn is not None and i in self.timbre_layers:
                r = timbre_fn(i, h)
                if r is not None:
                    h = h + r
            use_local = self.local_every and (i % self.local_every == 0)
            h = _run(blk, self.grad_ckpt, h, c, rope, local_mask if use_local else global_mask)
            if i < half:
                skips.append(h)
            if i in self.repa_layers:
                repa[i] = h
        return h, repa


def _down2(x: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] % 2:
        x = F.pad(x, (0, 0, 0, 1))
        valid = F.pad(valid, (0, 1))
    w = valid.float()[..., None]
    xs = (x * w).view(x.shape[0], -1, 2, x.shape[-1]).sum(2)
    ws = w.view(w.shape[0], -1, 2, 1).sum(2)
    return xs / ws.clamp_min(1.0), valid.view(valid.shape[0], -1, 2).any(-1)


def _up2(x: torch.Tensor, length: int) -> torch.Tensor:
    return x.repeat_interleave(2, dim=1)[:, :length]


class MultiRateUNet(nn.Module):
    def __init__(self, dim: int, heads: int, stage_blocks: tuple[int, ...] = (2, 2, 4, 2, 2),
                 stage_rates: tuple[int, ...] = (1, 2, 4, 2, 1), mlp_mult: float = 4.0,
                 local_window: int = 64, repa_stages: tuple[int, ...] = (1,)):
        super().__init__()
        assert len(stage_blocks) == len(stage_rates)
        self.dim, self.heads = dim, heads
        self.rates = list(stage_rates)
        self.stages = nn.ModuleList(
            nn.ModuleList(AdaLNBlock(dim, heads, mlp_mult) for _ in range(n)) for n in stage_blocks
        )
        self.bypass = nn.ParameterList(nn.Parameter(torch.full((dim,), 0.5)) for _ in stage_blocks)
        self.local_window = local_window
        self.repa_stages = tuple(repa_stages)
        # Number of full-rate blocks gets timbre attention (index into a flat list).
        self.full_rate_blocks = sum(n for n, r in zip(stage_blocks, stage_rates) if r == 1)
        self.grad_ckpt = False

    def forward(self, h: torch.Tensor, c: torch.Tensor, valid: torch.Tensor,
                timbre_fn: Callable[[int, torch.Tensor], torch.Tensor | None] | None = None):
        cache = {1: (h, valid)}
        cur_rate = 1
        repa = {}
        full_idx = 0
        stage_inputs: dict[int, torch.Tensor] = {}
        for si, (rate, blocks) in enumerate(zip(self.rates, self.stages)):
            # Move to this stage's rate.
            while cur_rate < rate:
                h, v = _down2(h, cache[cur_rate][1])
                cur_rate *= 2
                cache[cur_rate] = (h, v)
            while cur_rate > rate:
                cur_rate //= 2
                skip, v = cache[cur_rate]
                h = _up2(h, skip.shape[1])
                b = self.bypass[si].to(h.dtype)
                h = h * (1 - b) + skip * b  # Zipformer-style learnable bypass
            valid_r = cache[cur_rate][1]
            stage_inputs[si] = h
            rope = rope_freqs(h.shape[1], self.dim // self.heads, h.device)
            local_mask = build_attn_mask(valid_r, self.local_window // rate if self.local_window else None)
            global_mask = build_attn_mask(valid_r)
            for bi, blk in enumerate(blocks):
                if rate == 1 and timbre_fn is not None:
                    r = timbre_fn(full_idx, h)
                    if r is not None:
                        h = h + r
                    full_idx += 1
                h = _run(blk, self.grad_ckpt, h, c, rope, local_mask if bi % 2 == 0 else global_mask)
            # Residual over the stage (bypass of the stage itself).
            b = self.bypass[si].to(h.dtype)
            h = stage_inputs[si] * (1 - b) + h * b if si > 0 else h
            cache[cur_rate] = (h, valid_r)
            if si in self.repa_stages:
                repa[si] = h if rate == 1 else h.repeat_interleave(rate, dim=1)[:, : valid.shape[1]]
        return h, repa
