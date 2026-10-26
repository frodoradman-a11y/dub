"""Stage D0: train DubVAE (24 kHz <-> 25 Hz latent) with GAN + mel + KL
(+ optional semantic regularization).

  torchrun --nproc_per_node=8 -m dubflow.train.train_vae --config configs/vae.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from .. import SAMPLE_RATE
from ..features.backends import SSLTeacher
from ..utils.audio import load_audio
from ..utils.checkpoint import load_checkpoint, save_checkpoint, unwrap
from ..utils.config import load_config, save_config
from ..utils.train_utils import (
    DistributedWeightedSampler,
    Logger,
    autocast,
    cleanup_distributed,
    seed_everything,
    setup_distributed,
    to_device,
)
from ..vae.discriminators import (
    Discriminators,
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
    kl_loss,
    multiscale_mel_loss,
)
from ..vae.model import build_vae


class SegmentDataset(Dataset):
    def __init__(self, manifest: str, segment_sec: float, hop: int):
        with open(manifest, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.seg = int(segment_sec * SAMPLE_RATE) // hop * hop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        wav = load_audio(self.rows[i]["audio"], SAMPLE_RATE)
        if wav.numel() > self.seg:
            s = random.randint(0, wav.numel() - self.seg)
            wav = wav[s : s + self.seg]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.seg - wav.numel()))
        peak = wav.abs().max().clamp_min(1e-4)
        gain = random.uniform(0.3, 0.95) / peak
        return {"wav": wav * gain}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    rank, world, device = setup_distributed()
    seed_everything(1234 + rank)
    out_dir = Path(cfg.train.out_dir)
    log = Logger(out_dir, rank == 0)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    vae = build_vae(cfg.model).to(device)
    disc = Discriminators(tuple(cfg.disc.periods), tuple(cfg.disc.stft_ffts), cfg.disc.channels).to(device)
    opt_g = torch.optim.AdamW(vae.parameters(), cfg.train.lr_g, betas=tuple(cfg.train.betas))
    opt_d = torch.optim.AdamW(disc.parameters(), cfg.train.lr_d, betas=tuple(cfg.train.betas))
    step = 0
    last = out_dir / "last.pt"
    if last.exists():
        ck = load_checkpoint(last)
        vae.load_state_dict(ck["model"])
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        log.log(step, {"resumed": 1})
    if world > 1:
        vae = DDP(vae, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
        disc = DDP(disc, device_ids=[device.index] if device.type == "cuda" else None)

    teacher = None
    sem_w = cfg.semantic.get("weight", 0.0)
    if cfg.model.get("semantic_dim", 0) and cfg.semantic.get("teacher", "none") != "none" and sem_w > 0:
        teacher = SSLTeacher(cfg.semantic.teacher, str(device), SAMPLE_RATE, cfg.semantic.get("layer", 16))

    ds = SegmentDataset(cfg.data.manifest, cfg.data.segment_sec, unwrap(vae).hop)
    sampler = DistributedWeightedSampler(torch.ones(len(ds)), rank, world, seed=step)
    dl = iter(DataLoader(ds, cfg.train.batch_size, sampler=sampler, num_workers=cfg.data.num_workers,
                         drop_last=True, pin_memory=device.type == "cuda"))
    L = cfg.loss
    prec = cfg.train.precision
    while step < cfg.train.max_steps:
        batch = to_device(next(dl), device)
        wav = batch["wav"]
        use_gan = step >= cfg.train.disc_start_step
        with autocast(device, prec):
            out = vae(wav)
        recon = out["recon"][..., : wav.shape[-1]].float()
        # ---- discriminator
        if use_gan:
            with autocast(device, prec):
                d_loss = discriminator_loss(disc(wav), disc(recon.detach()))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.train.grad_clip)
            opt_d.step()
        # ---- generator
        mel = multiscale_mel_loss(recon, wav, SAMPLE_RATE)
        kl = kl_loss(out["mu"].float(), out["logvar"].float())
        g_loss = L.mel * mel + L.kl * kl
        logs = {"mel": mel.item(), "kl": kl.item()}
        if teacher is not None and "semantic" in out:
            tgt = teacher(wav, out["semantic"].shape[1]).float()
            tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.std(-1, keepdim=True) + 1e-5)
            sem = torch.nn.functional.mse_loss(out["semantic"].float(), tgt)
            g_loss = g_loss + sem_w * sem
            logs["sem"] = sem.item()
        if use_gan:
            with autocast(device, prec):
                fake_outs = disc(recon)
                real_outs = disc(wav)
            adv = generator_adv_loss(fake_outs)
            fm = feature_matching_loss(real_outs, fake_outs)
            g_loss = g_loss + L.adv * adv + L.feat * fm
            logs.update(adv=adv.item(), fm=fm.item(), d=d_loss.item())
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.train.grad_clip)
        opt_g.step()
        step += 1
        if step % cfg.train.log_every == 0:
            logs["gnorm"] = float(gn)
            log.log(step, logs)
        if rank == 0 and (step % cfg.train.save_every == 0 or step == cfg.train.max_steps):
            payload = dict(config=cfg.to_dict(), model=unwrap(vae).state_dict(), disc=unwrap(disc).state_dict(),
                           opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(), step=step)
            save_checkpoint(last, **payload)
            save_checkpoint(out_dir / f"vae_{step:07d}.pt", config=cfg.to_dict(), model=unwrap(vae).state_dict(), step=step)
    cleanup_distributed()


if __name__ == "__main__":
    main()
