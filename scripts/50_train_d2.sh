#!/usr/bin/env bash
# D2: dubbing fine-tune (EN->FA and FA->EN pairs) from the D1 checkpoint.
set -euo pipefail; source scripts/env.sh
$TORCHRUN -m dubflow.train.train_dubflow --config $CFG --vae $VAE_CKPT data.stage=dub \
  data.dub_manifest=data/dub/pairs.jsonl train.init_from=$RUN/d1/last.pt train.out_dir=$RUN/d2 \
  train.lr=3.0e-5 train.warmup_steps=1000 train.max_steps=40000 "$@"
