#!/usr/bin/env bash
# D3: Flow-GRPO with automatic rewards. Set grpo.asr_model=<persian whisper> to add the CER reward.
set -euo pipefail; source scripts/env.sh
$TORCHRUN -m dubflow.train.train_grpo --config $CFG --init $RUN/d2/last.pt --vae $VAE_CKPT \
  data.stage=dub data.dub_manifest=data/dub/pairs.jsonl train.out_dir=$RUN/d3 train.log_every=1 train.save_every=50 "$@"
