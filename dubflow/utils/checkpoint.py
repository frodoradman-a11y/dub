"""Checkpoint helpers. Every checkpoint stores the config it was built from so
models can be rebuilt without the original YAML."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Config, _wrap


def save_checkpoint(path: str | Path, **payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_vae(path: str | Path, device="cpu"):
    """DubVAE checkpoint (.pt), or AuK-VAE: "auk" (download from HuggingFace), a
    `vae.safetensors` file, or a directory containing one."""
    from ..vae.auk_vae import is_auk_spec, load_auk_vae
    from ..vae.model import build_vae

    if is_auk_spec(path):
        return load_auk_vae(str(path), device)
    ck = load_checkpoint(path)
    cfg = _wrap(ck["config"])
    vae = build_vae(cfg.model if "model" in cfg else cfg)
    state = ck.get("ema") or ck["model"]
    vae.load_state_dict(state, strict=False)
    if "latent_stats" in ck:
        vae.latent_mean.copy_(ck["latent_stats"]["mean"])
        vae.latent_std.copy_(ck["latent_stats"]["std"])
    return vae.to(device).eval()


def check_vae_compat(vae, latent_dim: int, stats: dict | None = None) -> None:
    """The flow model, its latent statistics and the VAE must come from the same
    latent space (DubVAE: 64 dims; AuK-VAE: 2 x 64 = 128 dims at 25 Hz)."""
    kind = getattr(vae, "kind", "dubvae")
    if int(vae.latent_dim) != int(latent_dim):
        raise ValueError(f"VAE '{kind}' has latent_dim {vae.latent_dim} but the model expects {latent_dim}: "
                         "use configs/dubflow_*_auk.yaml (128) with AuK-VAE and dubflow_m/s.yaml (64) with DubVAE")
    want = (stats or {}).get("vae_kind")
    if want is not None and want != kind:
        raise ValueError(f"latent statistics were computed with VAE '{want}' but '{kind}' was given")


def load_dubflow(path: str | Path, device="cpu", use_ema: bool = True):
    from ..model.dubflow import DubFlow

    ck = load_checkpoint(path)
    cfg = _wrap(ck["config"])
    model = DubFlow(cfg.model)
    state = ck.get("ema") if use_ema and ck.get("ema") is not None else ck["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def unwrap(model):
    return model.module if hasattr(model, "module") else model


__all__ = ["save_checkpoint", "load_checkpoint", "load_vae", "check_vae_compat", "load_dubflow", "unwrap", "Config"]
