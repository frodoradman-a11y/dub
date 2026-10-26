#!/usr/bin/env bash
# D0: DubVAE. Needs data/vae_manifest.jsonl ({"id","audio"} per line).
set -euo pipefail; source scripts/env.sh
$TORCHRUN -m dubflow.train.train_vae --config configs/vae.yaml train.out_dir=runs/vae "$@"
