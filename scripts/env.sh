# Common settings for all stage scripts. Edit to match your server.
export NGPU=${NGPU:-8}
export CFG=${CFG:-configs/dubflow_m.yaml}       # or configs/dubflow_s.yaml
export RUN=${RUN:-runs/m}                      # output root for this model size
export VAE_CKPT=${VAE_CKPT:-runs/vae/last.pt}
export SPK_BACKEND=${SPK_BACKEND:-wavlm_sv}
export EMO_BACKEND=${EMO_BACKEND:-emotion2vec}
export TORCHRUN="torchrun --nproc_per_node=${NGPU}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# Source augmentation (degraded copies of the conditioning audio, see data.src_aug_*).
# AUG must equal data.src_aug_versions in the config; NOISE_LIST / MUSIC_LIST are optional
# JSONL lists ({"audio": path}) of real noise and music & effects recordings.
export AUG=${AUG:-2}
AUG_ARGS="--augment $AUG"
[ -n "${NOISE_LIST:-}" ] && AUG_ARGS="$AUG_ARGS --noise-list $NOISE_LIST"
[ -n "${MUSIC_LIST:-}" ] && AUG_ARGS="$AUG_ARGS --music-list $MUSIC_LIST"
export AUG_ARGS
