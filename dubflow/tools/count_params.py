"""Print parameter counts for a DubFlow or DubVAE config (no weights allocated).

  dubflow-params configs/dubflow_m.yaml
"""

from __future__ import annotations

import argparse

import torch

from ..utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    with torch.device("meta"):
        if "backbone" in cfg.model:
            from ..model.dubflow import DubFlow, count_parameters

            counts = count_parameters(DubFlow(cfg.model))
        else:
            from ..vae.model import build_vae

            vae = build_vae(cfg.model)
            counts = {"encoder": sum(p.numel() for p in vae.encoder.parameters()) / 1e6,
                      "decoder": sum(p.numel() for p in vae.decoder.parameters()) / 1e6,
                      "total": sum(p.numel() for p in vae.parameters()) / 1e6}
    for k, v in counts.items():
        print(f"{k:>14}: {v:8.1f} M")


if __name__ == "__main__":
    main()
