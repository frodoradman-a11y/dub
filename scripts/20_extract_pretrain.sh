#!/usr/bin/env bash
# Feature cache for pretraining data (one shard per GPU), then index + merged stats.
set -euo pipefail; source scripts/env.sh
MAN=${MAN:-data/pretrain_manifest.jsonl}; OUT=${OUT:-feats/pretrain}
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i python -m dubflow.preprocess.extract --manifest $MAN --vae $VAE_CKPT --out $OUT \
    --speaker-backend $SPK_BACKEND --emotion-backend $EMO_BACKEND --shard $i --num-shards $NGPU --skip-existing $AUG_ARGS &
done
wait
python -m dubflow.preprocess.build_index --manifest $MAN --cache $OUT --out $OUT/index.pt
