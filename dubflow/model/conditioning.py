"""Conditioning modules: text encoder, time-anchored upsampling, synchronous
tracks, and the band-limited timbre-window cross-attention (TSVB)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock, build_attn_mask, rope_freqs


class TextEncoder(nn.Module):
    """Character-level transformer with dual language injection (FiLM on embeddings)."""

    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, n_langs: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.lang_film = nn.Embedding(n_langs, 2 * dim)
        nn.init.zeros_(self.lang_film.weight)
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads) for _ in range(layers))
        self.norm = RMSNorm(dim)
        self.heads = heads

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        gamma, beta = self.lang_film(lang)[:, None].chunk(2, -1)
        x = x * (1 + gamma) + beta
        rope = rope_freqs(tokens.shape[1], x.shape[-1] // self.heads, x.device)
        attn_mask = build_attn_mask(mask)
        for blk in self.blocks:
            x = blk(x, rope, attn_mask)
        return self.norm(x) * mask[..., None]


def anchored_upsample(text_feats: torch.Tensor, frame_tok: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
    """Gather token features per frame.

    text_feats [B, N, d]; frame_tok [B, S, T] long (token index per frame and
    text stream, -1 = no text); null [d] embedding for frames without text.
    Returns [B, S, T, d].
    """
    b, s, t = frame_tok.shape
    idx = frame_tok.clamp_min(0)
    gathered = torch.gather(text_feats[:, None].expand(b, s, -1, -1), 2, idx[..., None].expand(-1, -1, -1, text_feats.shape[-1]))
    has = (frame_tok >= 0)[..., None]
    return torch.where(has, gathered, null.to(gathered.dtype))


class TrackProjector(nn.Module):
    """Projects synchronous frame tracks (identity, prosody, emotion, activity).

    Each named group gets its own dropout during training so the model learns
    to work with any subset (and so CFG can drop them).
    """

    def __init__(self, groups: dict[str, int], dim: int):
        super().__init__()
        self.groups = dict(groups)
        self.proj = nn.ModuleDict({k: nn.Linear(v, dim) for k, v in self.groups.items()})
        self.norm = RMSNorm(dim)

    @property
    def total_dim(self) -> int:
        return sum(self.groups.values())

    def forward(self, tracks: torch.Tensor, group_keep: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        out = 0.0
        off = 0
        for name, n in self.groups.items():
            part = self.proj[name](tracks[..., off : off + n])
            if group_keep is not None and name in group_keep:
                part = part * group_keep[name][:, None, None].to(part.dtype)
            out = out + part
            off += n
        return self.norm(out)


class SourceEncoder(nn.Module):
    """Light conv encoder over the source (original) latent for timbre windows."""

    def __init__(self, latent_dim: int, dim: int, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(latent_dim, dim)
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, 5, padding=2, groups=1) for _ in range(layers))
        self.norm = RMSNorm(dim)

    def forward(self, src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.inp(src)
        for conv in self.convs:
            h = h + F.gelu(conv(h.transpose(1, 2)).transpose(1, 2))
        return self.norm(h) * mask[..., None]


def gather_windows(src_feats: torch.Tensor, src_mask: torch.Tensor, centers: torch.Tensor, half: int, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """For each target frame gather source frames [c-half, c+half] (step `stride`).

    src_feats [B, Ts, dk]; src_mask [B, Ts]; centers [B, T] (-1 = no window).
    Returns windows [B, T, W, dk] and valid [B, T, W].
    """
    b, ts, dk = src_feats.shape
    offs = torch.arange(-half, half + 1, stride, device=src_feats.device)
    pos = centers[..., None] + offs  # [B, T, W]
    valid = (centers[..., None] >= 0) & (pos >= 0) & (pos < ts)
    pos_c = pos.clamp(0, ts - 1)
    valid = valid & torch.gather(src_mask[:, None].expand(-1, pos.shape[1], -1), 2, pos_c)
    win = torch.gather(src_feats[:, None].expand(-1, pos.shape[1], -1, -1), 2, pos_c[..., None].expand(-1, -1, -1, dk))
    return win, valid


class TimbreWindowAttention(nn.Module):
    """Cross-attention from each target frame to its own time-local source window.

    A learned null key/value is always available, so frames without a window
    (or with dropped windows) attend to it instead of producing NaNs.
    """

    def __init__(self, dim: int, src_dim: int, heads: int = 4, inner: int = 256):
        super().__init__()
        self.heads, self.hd = heads, inner // heads
        self.norm = RMSNorm(dim)
        self.q = nn.Linear(dim, inner, bias=False)
        self.kv = nn.Linear(src_dim, 2 * inner, bias=False)
        self.null_kv = nn.Parameter(torch.zeros(2, inner))
        self.out = nn.Linear(inner, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, win: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, t, w, _ = win.shape
        q = self.q(self.norm(x)).view(b, t, self.heads, 1, self.hd)
        k, v = self.kv(win).view(b, t, w, 2, self.heads, self.hd).unbind(3)
        nk, nv = self.null_kv.view(2, 1, 1, 1, self.heads, self.hd).to(k.dtype).unbind(0)
        k = torch.cat([nk.expand(b, t, 1, -1, -1), k], 2).transpose(2, 3)  # [B,T,H,W+1,hd]
        v = torch.cat([nv.expand(b, t, 1, -1, -1), v], 2).transpose(2, 3)
        mask = torch.cat([torch.ones_like(valid[..., :1]), valid], -1)[:, :, None, None, :]  # [B,T,1,1,W+1]
        att = (q * self.hd ** -0.5) @ k.transpose(-1, -2)  # [B,T,H,1,W+1]
        att = att.masked_fill(~mask, torch.finfo(att.dtype).min).softmax(-1)
        y = (att @ v).reshape(b, t, -1)
        return self.out(y)
