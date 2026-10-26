# DubFlow

Time-synchronous flow-matching model for **scene-level EN ⇄ FA dubbing**.
It re-performs the original actor's line in the target language: same voice,
same moment, same feeling, new text, with **no speaker diarization**.

- **DubFlow-M** (quality): ~0.8B DiT backbone + DubVAE (~0.17B)
- **DubFlow-S** (fast): ~0.2B multi-rate U-Net backbone + DubVAE
- 24 kHz output, 25 Hz continuous latent, scenes up to 60 s, 1–4 step student

Persian documentation: [`docs/implementation-guide-fa.md`](docs/implementation-guide-fa.md) (full guide),
[`docs/dubflow-plan-b-fa.md`](docs/dubflow-plan-b-fa.md) (design / proposal).

## Quick start

```bash
bash scripts/00_setup.sh              # venv + deps + test-suite (CPU, ~1 min)
dubflow-params configs/dubflow_m.yaml # parameter budget per module
```

Training stages (8 GPUs, edit `scripts/env.sh`):

| stage | script | what |
|---|---|---|
| D0 | `scripts/10_train_vae.sh` | DubVAE (24 kHz ⇄ 25 Hz × 64) |
| — | `scripts/20_extract_pretrain.sh` | feature cache + index for the 50k-hour corpus |
| D1 | `scripts/30_train_d1.sh` | pretraining (synthetic multi-speaker scenes, offset-window) |
| — | `scripts/40_build_dub.sh` | scene pairs from aligned films (+ features) |
| D2 | `scripts/50_train_d2.sh` | dubbing fine-tune (EN→FA and FA→EN) |
| D3 | `scripts/60_train_d3_grpo.sh` | Flow-GRPO with automatic constrained rewards |
| D4 | `scripts/70_distill.sh` | 4-step / 1-step student (M→M or M→S) |
| — | `scripts/80_dub.sh` | dub a film: vocals + script → dubbed vocals (+ M&E mix) |

Use `CFG=configs/dubflow_s.yaml RUN=runs/s` for the small model.

## Layout

```
dubflow/
  vae/          DubVAE + GAN discriminators/losses
  model/        DubFlow, backbones (DiT / multi-rate U-Net), conditioning (TSVB)
  data/         datasets (pretrain / dub), tracks, time-anchored upsampling
  features/     speaker / emotion / SSL-teacher backends (+ test fallbacks)
  preprocess/   extract, index, build_dub_pairs, forced alignment
  train/        train_vae, train_dubflow (D1/D2), train_grpo (D3), distill (D4)
  rl/           Flow-GRPO (SDE rollouts, constrained reward fusion)
  infer/        scene splitting and the dubbing CLI
  eval/         metrics (SIM, SSP, boundary, CER/WER, SLC, pause IoU)
configs/        vae, dubflow_m, dubflow_s, tiny test configs
scripts/        stage launch scripts
tests/          unit + end-to-end tests (CPU, synthetic data)
```
