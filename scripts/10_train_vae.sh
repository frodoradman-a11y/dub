#!/usr/bin/env bash
# D0: DubVAE. Needs data/vae_manifest.jsonl ({"id","audio"} per line).
set -euo pipefail; source scripts/env.sh
case "$VAE_CKPT" in
  auk|auk:*|*.safetensors) echo "VAE_CKPT=$VAE_CKPT is AuK-VAE: nothing to train (skip D0)"; exit 0;;
esac
$TORCHRUN -m dubflow.train.train_vae --config configs/vae.yaml train.out_dir=runs/vae "$@"
