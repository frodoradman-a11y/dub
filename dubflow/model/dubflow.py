"""DubFlow: conditional flow matching for time-synchronous scene dubbing.

Convention (rectified flow): x_t = (1 - t) * x0 + t * x1, target velocity
v = x1 - x0, integrate t: 0 -> 1. x1 is the normalized target latent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..text.tokenizer import LANGS, TextTokenizer
from .backbones import DiTBackbone, MultiRateUNet
from .conditioning import (
    SourceEncoder,
    TextEncoder,
    TimbreWindowAttention,
    TrackProjector,
    anchored_upsample,
    gather_windows,
)
from .layers import RMSNorm, TimestepEmbedding


@dataclass
class Drop:
    """Per-sample keep masks (1 = keep) used for CFG training and guidance."""

    text: torch.Tensor
    timbre: torch.Tensor
    tracks: dict[str, torch.Tensor]


class DubFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg
        self.cfg = cfg
        self.latent_dim = m.latent_dim
        dim = m.dim
        self.dim = dim
        tok = TextTokenizer()
        self.text_encoder = TextEncoder(tok.vocab_size, m.text.dim, m.text.layers, m.text.heads, len(LANGS))
        self.text_proj = nn.Linear(m.text.dim, dim)
        self.null_text = nn.Parameter(torch.zeros(dim))
        self.n_streams = m.get("n_streams", 2)
        self.stream_emb = nn.Parameter(torch.zeros(self.n_streams, dim))
        self.track_groups = dict(m.tracks)
        self.track_proj = TrackProjector(self.track_groups, dim)
        self.in_proj = nn.Linear(m.latent_dim, dim)
        self.t_embed = TimestepEmbedding(dim)
        self.lang_embed = nn.Embedding(len(LANGS), dim)
        nn.init.zeros_(self.lang_embed.weight)
        self.step_embed = TimestepEmbedding(dim) if m.get("step_cond", False) else None

        bb = m.backbone
        if bb.type == "dit":
            self.backbone = DiTBackbone(dim, bb.depth, bb.heads, bb.get("mlp_mult", 4.0),
                                        bb.get("local_window", 64), bb.get("local_every", 2),
                                        m.timbre.get("every", 2) if m.timbre.enabled else 0,
                                        tuple(m.repa.layers) if m.repa.enabled else ())
            timbre_slots = list(self.backbone.timbre_layers)
        elif bb.type == "unet":
            self.backbone = MultiRateUNet(dim, bb.heads, tuple(bb.stage_blocks), tuple(bb.stage_rates),
                                          bb.get("mlp_mult", 4.0), bb.get("local_window", 64),
                                          tuple(m.repa.layers) if m.repa.enabled else ())
            every = m.timbre.get("every", 2)
            timbre_slots = [i for i in range(self.backbone.full_rate_blocks) if every and i % every == 0]
        else:
            raise ValueError(f"unknown backbone type {bb.type}")

        self.timbre_enabled = bool(m.timbre.enabled)
        if self.timbre_enabled:
            self.src_encoder = SourceEncoder(m.latent_dim, m.timbre.src_dim, m.timbre.get("src_layers", 2))
            self.timbre_attn = nn.ModuleList(
                TimbreWindowAttention(dim, m.timbre.src_dim, m.timbre.heads, m.timbre.inner) for _ in timbre_slots
            )
            self.timbre_slot_index = {s: j for j, s in enumerate(timbre_slots)}
            self.window_half = m.timbre.window_half
            self.window_stride = m.timbre.get("window_stride", 1)

        self.out_norm = RMSNorm(dim, affine=False)
        self.out_ada = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.out_ada.weight)
        nn.init.zeros_(self.out_ada.bias)
        self.out = nn.Linear(dim, m.latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.repa_enabled = bool(m.repa.enabled)
        if self.repa_enabled:
            self.repa_proj = nn.ModuleDict({
                str(k): nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, m.repa.target_dim))
                for k in m.repa.layers
            })
        self.drop_cfg = m.get("cfg_dropout", {})

    # ------------------------------------------------------------------ conditions
    def sample_drop(self, b: int, device) -> Drop:
        d = self.drop_cfg

        def keep(p: float) -> torch.Tensor:
            return (torch.rand(b, device=device) >= p).float()

        all_keep = keep(d.get("all", 0.1))
        tracks = {name: keep(d.get(f"track_{name}", d.get("tracks", 0.2))) * all_keep for name in self.track_groups}
        return Drop(text=keep(d.get("text", 0.1)) * all_keep, timbre=keep(d.get("timbre", 0.2)) * all_keep, tracks=tracks)

    def full_keep(self, b: int, device) -> Drop:
        one = torch.ones(b, device=device)
        return Drop(text=one, timbre=one, tracks={k: one for k in self.track_groups})

    def prepare(self, batch: dict, drop: Drop) -> dict:
        """Compute step-invariant conditioning once (reused across ODE steps)."""
        b, t = batch["frame_mask"].shape
        text = self.text_proj(self.text_encoder(batch["text"], batch["text_mask"], batch["lang"]))
        frame_tok = batch["frame_tok"].clone()
        frame_tok[drop.text < 0.5] = -1
        up = anchored_upsample(text, frame_tok, self.null_text)  # [B,S,T,d]
        has = (frame_tok >= 0)[..., None].to(up.dtype)
        up = up + self.stream_emb[: up.shape[1], None].to(up.dtype) * has
        cond_seq = up.sum(1) + self.track_proj(batch["tracks"], drop.tracks)
        out = {"cond_seq": cond_seq, "valid": batch["frame_mask"], "lang": batch["lang"]}
        if self.timbre_enabled:
            src = self.src_encoder(batch["src_latent"], batch["src_mask"])
            win, valid = gather_windows(src, batch["src_mask"], batch["win_center"], self.window_half, self.window_stride)
            valid = valid & (drop.timbre[:, None, None] > 0.5)
            out["win"], out["win_valid"] = win, valid
        return out

    # ------------------------------------------------------------------ network
    def _timbre(self, layer: int, h: torch.Tensor, cond: dict) -> torch.Tensor | None:
        j = self.timbre_slot_index.get(layer)
        return None if j is None else self.timbre_attn[j](h, cond["win"], cond["win_valid"])

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond: dict, step: torch.Tensor | None = None):
        h = self.in_proj(x_t) + cond["cond_seq"]
        c = self.t_embed(t) + self.lang_embed(cond["lang"])
        if self.step_embed is not None:
            c = c + self.step_embed(step if step is not None else torch.zeros_like(t))
        timbre_fn = (lambda i, hh: self._timbre(i, hh, cond)) if self.timbre_enabled else None
        h, repa = self.backbone(h, c, cond["valid"], timbre_fn)
        shift, scale = self.out_ada(F.silu(c))[:, None].chunk(2, -1)
        v = self.out(self.out_norm(h) * (1 + scale) + shift)
        return v * cond["valid"][..., None].to(v.dtype), repa

    # ------------------------------------------------------------------ training
    def sample_t(self, b: int, device) -> torch.Tensor:
        ts = self.cfg.get("t_sampling", {"type": "logit_normal", "mean": 0.0, "std": 1.0})
        if ts.get("type", "logit_normal") == "uniform":
            return torch.rand(b, device=device)
        return torch.sigmoid(torch.randn(b, device=device) * ts.get("std", 1.0) + ts.get("mean", 0.0))

    def forward(self, batch: dict) -> dict:
        x1 = batch["x1"]
        b = x1.shape[0]
        drop = self.sample_drop(b, x1.device)
        cond = self.prepare(batch, drop)
        t = self.sample_t(b, x1.device)
        x0 = torch.randn_like(x1)
        tt = t[:, None, None]
        x_t = (1 - tt) * x0 + tt * x1
        v, repa = self.velocity(x_t, t, cond)
        target = x1 - x0
        m = (batch["frame_mask"] & batch["loss_mask"]).to(v.dtype)[..., None]
        denom = m.sum().clamp_min(1.0) * v.shape[-1]
        fm = ((v.float() - target.float()) ** 2 * m).sum() / denom
        losses = {"fm": fm}
        total = fm
        if self.repa_enabled and "repa_target" in batch and repa:
            rl = 0.0
            tgt = F.normalize(batch["repa_target"].float(), dim=-1)
            mm = m[..., 0]
            for k, feat in repa.items():
                pred = F.normalize(self.repa_proj[str(k)](feat).float(), dim=-1)
                cos = (pred * tgt).sum(-1)
                rl = rl + ((1 - cos) * mm).sum() / mm.sum().clamp_min(1.0)
            rl = rl / len(repa)
            losses["repa"] = rl
            total = total + self.cfg.repa.weight * rl
        losses["loss"] = total
        return losses

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample(self, batch: dict, steps: int = 16, guidance: dict | None = None, sway: float = -1.0,
               seed: int | None = None, x0: torch.Tensor | None = None, step_cond: bool = False) -> torch.Tensor:
        """Euler ODE sampling with multi-condition CFG.

        guidance: {"all": s_all, "text": s_text, "timbre": s_timbre, "tracks": s_tracks}
        v = v_c + sum_k s_k * (v_c - v_{drop k})
        """
        valid = batch["frame_mask"]
        b, t = valid.shape
        dev = valid.device
        gen = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
        x = x0 if x0 is not None else torch.randn(b, t, self.latent_dim, device=dev, generator=gen)
        full = self.full_keep(b, dev)
        cond = self.prepare(batch, full)
        guidance = {k: v for k, v in (guidance or {}).items() if v}
        variants = {}
        zero = torch.zeros(b, device=dev)
        for k in guidance:
            if k == "all":
                d = Drop(text=zero, timbre=zero, tracks={n: zero for n in self.track_groups})
            elif k == "text":
                d = Drop(text=zero, timbre=full.timbre, tracks=full.tracks)
            elif k == "timbre":
                d = Drop(text=full.text, timbre=zero, tracks=full.tracks)
            elif k == "tracks":
                d = Drop(text=full.text, timbre=full.timbre, tracks={n: zero for n in self.track_groups})
            else:
                raise ValueError(k)
            variants[k] = self.prepare(batch, d)
        ts = torch.linspace(0, 1, steps + 1, device=dev)
        if sway:
            ts = ts + sway * (torch.cos(math.pi / 2 * ts) - 1 + ts)
        for i in range(steps):
            t0, t1 = ts[i], ts[i + 1]
            tb = t0.expand(b)
            sb = (t1 - t0).expand(b) if step_cond else None
            v, _ = self.velocity(x, tb, cond, sb)
            if variants:
                vg = v.clone()
                for k, vc in variants.items():
                    vd, _ = self.velocity(x, tb, vc, sb)
                    vg = vg + guidance[k] * (v - vd)
                v = vg
            x = x + (t1 - t0) * v
        return x * valid[..., None].to(x.dtype)


def build_dubflow(cfg) -> DubFlow:
    return DubFlow(cfg)


def count_parameters(model: nn.Module) -> dict[str, float]:
    groups = {
        "text_encoder": model.text_encoder,
        "backbone": model.backbone,
        "track_proj": model.track_proj,
    }
    if getattr(model, "timbre_enabled", False):
        groups["timbre"] = nn.ModuleList([model.src_encoder, model.timbre_attn])
    out = {k: sum(p.numel() for p in v.parameters()) / 1e6 for k, v in groups.items()}
    out["total"] = sum(p.numel() for p in model.parameters()) / 1e6
    return out
