"""Stages D1 (pretraining) and D2 (dubbing fine-tune) for DubFlow-S / DubFlow-M.

  # D1 pretraining
  torchrun --nproc_per_node=8 -m dubflow.train.train_dubflow --config configs/dubflow_m.yaml
  # D2 dubbing fine-tune from the D1 checkpoint
  torchrun --nproc_per_node=8 -m dubflow.train.train_dubflow --config configs/dubflow_m.yaml \
      data.stage=dub train.init_from=runs/dubflow_m/last.pt train.out_dir=runs/dubflow_m_d2 \
      train.lr=3e-5 train.max_steps=30000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .. import SAMPLE_RATE
from ..data.datasets import DubDataset, PretrainDataset, collate
from ..features.backends import SSLTeacher
from ..model.dubflow import DubFlow, count_parameters
from ..preprocess.build_index import merge_latent_stats
from ..utils.checkpoint import load_checkpoint, save_checkpoint, unwrap
from ..utils.config import load_config, save_config
from ..utils.train_utils import (
    EMA,
    DistributedWeightedSampler,
    Logger,
    all_reduce_mean,
    autocast,
    cleanup_distributed,
    seed_everything,
    setup_distributed,
    to_device,
    warmup_lambda,
)


def build_dataset(cfg, stats: dict, return_wav: bool, seed: int):
    d = cfg.data
    n_streams = cfg.model.get("n_streams", 2)
    if d.stage == "pretrain":
        return PretrainDataset(d.manifest, d.cache, d.index, stats, d, n_streams, return_wav, seed)
    if d.stage == "dub":
        return DubDataset(d.dub_manifest, d.dub_src_cache, d.dub_tgt_cache, stats, d, n_streams, return_wav, seed=seed)
    raise ValueError(f"unknown data.stage {d.stage}")


def load_stats(cfg) -> dict:
    """Latent normalization must stay identical across stages: reuse the stats of
    the checkpoint we initialize/resume from; otherwise compute them from the cache."""
    stats = None
    last = Path(cfg.train.out_dir) / "last.pt"
    for src in (last if last.exists() else None, cfg.train.get("init_from")):
        if src:
            stats = load_checkpoint(src).get("latent_stats")
            if stats is not None:
                break
    if stats is None:
        cache = cfg.data.cache if cfg.data.stage == "pretrain" else cfg.data.dub_tgt_cache
        stats = merge_latent_stats(cache)
    if int(stats["mean"].numel()) != int(cfg.model.latent_dim):
        raise ValueError(f"feature cache latents are {stats['mean'].numel()}-dim (VAE '{stats.get('vae_kind', 'dubvae')}') "
                         f"but model.latent_dim={cfg.model.latent_dim}: use configs/dubflow_*_auk.yaml for AuK-VAE")
    dims = {"spk_island": stats["speaker_dim"], "spk_short": stats["speaker_dim"], "spk_overlap": stats["speaker_dim"],
            "emotion": stats["emotion_dim"]}
    for k, v in dims.items():
        if int(cfg.model.tracks[k]) != int(v):
            raise ValueError(f"model.tracks.{k}={cfg.model.tracks[k]} but feature cache has {v}; fix the config")
    return stats


@torch.no_grad()
def reencode_overlaps(batch: dict, vae, stats: dict) -> None:
    """Synthetic overlapping-speech scenes exist only as mixed waveforms: encode
    them with the (frozen) VAE in place of the placeholder latents. In
    pretraining the conditioning source is the scene itself, so src_latent too."""
    sel = batch["reencode"]
    if "wav" not in batch or not bool(sel.any()):
        return
    t = batch["x1"].shape[1]
    _, mu, _ = vae.encode(batch["wav"][sel].float())
    z = mu.transpose(1, 2).float()[:, :t]
    z = torch.nn.functional.pad(z, (0, 0, 0, t - z.shape[1]))
    z = (z - stats["mean"].to(z.device)) / stats["std"].to(z.device)
    x1 = torch.where(batch["frame_mask"][sel][..., None], z, batch["x1"][sel])
    batch["x1"][sel] = x1
    ts = batch["src_latent"].shape[1]
    batch["src_latent"][sel] = x1[:, :ts]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--vae", default=None,
                    help="DubVAE checkpoint or AuK-VAE ('auk' / vae.safetensors): audio samples, and encoding of "
                         "synthetic overlap scenes (required when data.overlap_p > 0 in pretraining)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    rank, world, device = setup_distributed()
    seed_everything(42 + rank)
    out_dir = Path(cfg.train.out_dir)
    log = Logger(out_dir, rank == 0)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    stats = load_stats(cfg)
    model = DubFlow(cfg.model).to(device)
    if rank == 0:
        log.log(0, {f"params/{k}": v for k, v in count_parameters(model).items()})
    unwrap(model).backbone.grad_ckpt = bool(cfg.train.get("activation_checkpointing", False))

    step = 0
    last = out_dir / "last.pt"
    ema = None
    opt = torch.optim.AdamW(model.parameters(), cfg.train.lr, betas=tuple(cfg.train.betas),
                            weight_decay=cfg.train.weight_decay)
    if last.exists():
        ck = load_checkpoint(last)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        ema = EMA(model, cfg.train.ema_decay)
        if ck.get("ema") is not None:
            ema.shadow.load_state_dict(ck["ema"])
        log.log(step, {"resumed": 1})
    elif cfg.train.get("init_from"):
        ck = load_checkpoint(cfg.train.init_from)
        src = ck.get("ema") or ck["model"]
        missing, unexpected = model.load_state_dict(src, strict=False)
        log.log(0, {"init_missing": len(missing), "init_unexpected": len(unexpected)})
    if ema is None:
        ema = EMA(model, cfg.train.ema_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_lambda(cfg.train.warmup_steps, cfg.train.max_steps,
                                                                  cfg.train.get("lr_decay", "constant")),
                                              last_epoch=step - 1 if step else -1)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
              find_unused_parameters=True) if world > 1 else model

    vae_path = args.vae or cfg.train.get("vae")
    overlap_vae = None
    if cfg.data.stage == "pretrain" and float(cfg.data.get("overlap_p", 0.0)) > 0:
        if not vae_path:
            raise ValueError("data.overlap_p > 0 needs the DubVAE to encode mixed overlap scenes: pass --vae "
                             "(or train.vae=...), or set data.overlap_p=0")
        from ..utils.checkpoint import check_vae_compat, load_vae

        overlap_vae = load_vae(vae_path, device)
        check_vae_compat(overlap_vae, cfg.model.latent_dim, stats)
        overlap_vae.requires_grad_(False)
    repa_on = bool(cfg.model.repa.enabled)
    teacher = SSLTeacher(cfg.repa_teacher.backend, str(device), SAMPLE_RATE, cfg.repa_teacher.get("layer", 16)) if repa_on else None
    ds = build_dataset(cfg, stats, return_wav=repa_on, seed=1000 + rank + step)
    weights = getattr(ds, "weights", torch.ones(len(ds)))
    sampler = DistributedWeightedSampler(weights, rank, world, seed=step)
    dl = iter(DataLoader(ds, cfg.train.batch_size, sampler=sampler, num_workers=cfg.data.num_workers,
                         collate_fn=collate, drop_last=True, pin_memory=device.type == "cuda",
                         persistent_workers=cfg.data.num_workers > 0))
    accum = int(cfg.train.get("grad_accum", 1))
    prec = cfg.train.precision
    model.train()
    while step < cfg.train.max_steps:
        opt.zero_grad(set_to_none=True)
        agg: dict[str, float] = {}
        for _ in range(accum):
            batch = to_device(next(dl), device)
            if overlap_vae is not None:
                reencode_overlaps(batch, overlap_vae, stats)
            wav, wav_ok = batch.pop("wav", None), batch.pop("wav_ok", None)
            if (teacher is not None and step < cfg.repa_teacher.get("until_step", 10**12) and wav is not None
                    and bool(wav_ok.all())):
                batch["repa_target"] = teacher(wav, batch["x1"].shape[1])
            with autocast(device, prec):
                losses = ddp(batch)
            (losses["loss"] / accum).backward()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(all_reduce_mean(v)) / accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()
        ema.update(unwrap(model), step)
        step += 1
        if step % cfg.train.log_every == 0:
            agg.update(gnorm=float(gn), lr=sched.get_last_lr()[0])
            log.log(step, agg)
        if rank == 0 and (step % cfg.train.save_every == 0 or step == cfg.train.max_steps):
            payload = dict(config=cfg.to_dict(), model=unwrap(model).state_dict(), ema=ema.state_dict(),
                           opt=opt.state_dict(), step=step,
                           latent_stats={k: stats[k] for k in ("mean", "std", "silence", "speaker_dim", "emotion_dim",
                                                               "vae_kind") if k in stats})
            save_checkpoint(last, **payload)
            payload.pop("opt")
            save_checkpoint(out_dir / f"dubflow_{step:07d}.pt", **payload)
            if vae_path:
                write_sample(unwrap(ema.shadow), vae_path, batch, cfg, stats, out_dir / "samples" / f"{step:07d}.wav", device)
    cleanup_distributed()


@torch.no_grad()
def write_sample(model, vae_path: str, batch: dict, cfg, stats: dict, path: Path, device) -> None:
    from ..utils.audio import save_audio
    from ..utils.checkpoint import load_vae

    vae = load_vae(vae_path, device)
    one = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in batch.items()}
    x = model.sample(one, steps=cfg.sample.steps, guidance=dict(cfg.sample.guidance), sway=cfg.sample.sway, seed=0)
    n = int(one["frame_mask"][0].sum())
    z = x[:, :n] * stats["std"].to(device) + stats["mean"].to(device)
    wav = vae.decode(z.transpose(1, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(str(path), wav[0], SAMPLE_RATE)


if __name__ == "__main__":
    main()
