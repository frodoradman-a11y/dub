# Common settings for all stage scripts. Edit to match your server.
export NGPU=${NGPU:-8}
export CFG=${CFG:-configs/dubflow_m.yaml}       # or configs/dubflow_s.yaml
export RUN=${RUN:-runs/m}                      # output root for this model size
export VAE_CKPT=${VAE_CKPT:-runs/vae/last.pt}
export SPK_BACKEND=${SPK_BACKEND:-wavlm_sv}
export EMO_BACKEND=${EMO_BACKEND:-emotion2vec}
export TORCHRUN="torchrun --nproc_per_node=${NGPU}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
