#!/usr/bin/env bash
# D1: pretraining on the large monolingual corpus (+ synthetic multi-speaker scenes).
set -euo pipefail; source scripts/env.sh
$TORCHRUN -m dubflow.train.train_dubflow --config $CFG --vae $VAE_CKPT data.stage=pretrain train.out_dir=$RUN/d1 "$@"
