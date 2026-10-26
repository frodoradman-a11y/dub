#!/usr/bin/env bash
# Inference: ./scripts/80_dub.sh vocals_en.wav script_fa.json out_fa.wav [music_effects.wav]
set -euo pipefail; source scripts/env.sh
MODEL=${MODEL:-$RUN/d4/last.pt}
BG=""; [ $# -ge 4 ] && BG="--bg $4"
python -m dubflow.infer.dub --model $MODEL --vae $VAE_CKPT --src "$1" --script "$2" --out "$3" $BG \
  --speaker-backend $SPK_BACKEND --emotion-backend $EMO_BACKEND
