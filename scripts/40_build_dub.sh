#!/usr/bin/env bash
# Dubbing pairs. FILMS_TSV lines: film_id <tab> src_vocals.wav <tab> tgt_vocals_vc.wav <tab> tgt_script.(json|srt)
set -euo pipefail; source scripts/env.sh
FILMS_TSV=${FILMS_TSV:-data/films.tsv}; OUT=${OUT:-data/dub}
while IFS=$'\t' read -r film src tgt script; do
  python -m dubflow.preprocess.build_dub_pairs --film "$film" --src-audio "$src" --tgt-audio "$tgt" \
    --tgt-script "$script" --src-lang en --tgt-lang fa --out $OUT --max-sec 30
done < "$FILMS_TSV"
for side in src tgt; do
  EXTRA=""; [ $side = src ] && EXTRA="--short-window $AUG_ARGS"   # degraded copies: source side only
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i python -m dubflow.preprocess.extract --manifest $OUT/${side}_manifest.jsonl --vae $VAE_CKPT \
      --out feats/dub_$side --speaker-backend $SPK_BACKEND --emotion-backend $EMO_BACKEND $EXTRA \
      --shard $i --num-shards $NGPU --skip-existing &
  done
  wait
  python -c "from dubflow.preprocess.build_index import merge_latent_stats as m; m('feats/dub_$side')"
done
