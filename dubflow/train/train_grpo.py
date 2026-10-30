"""Stage D3: Flow-GRPO post-training with automatic, constrained rewards.

Rewards (all label-free except optional CER):
  sim       speaker similarity per source speech island (output vs. original)
  ssp       speaker-structure preservation (pairwise identity correlation)
  boundary  share of output speech outside the script chunks (hard gate)
  cer       optional ASR character error rate (hard gate + soft term)

  torchrun --nproc_per_node=8 -m dubflow.train.train_grpo --config configs/dubflow_m.yaml \
      --init runs/dubflow_m_d2/last.pt --vae runs/vae/vae_0800000.pt train.out_dir=runs/dubflow_m_d3
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
import torch.distributed as dist

from .. import FRAME_RATE, LATENT_HOP, SAMPLE_RATE
from ..data.datasets import DubDataset, collate
from ..eval.metrics import ASR, boundary_error, cer, island_similarity, ssp
from ..features.backends import SpeakerEmbedder
from ..model.dubflow import DubFlow
from ..rl.flow_grpo import combine_rewards, group_advantages, grpo_loss, rollout
from ..utils.checkpoint import check_vae_compat, load_checkpoint, load_vae, save_checkpoint, unwrap
from ..utils.config import load_config, save_config
from ..utils.train_utils import Logger, cleanup_distributed, seed_everything, setup_distributed, to_device


def repeat_batch(batch: dict, g: int) -> dict:
    return {k: (v.repeat_interleave(g, 0) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def decode(vae, x: torch.Tensor, stats: dict) -> torch.Tensor:
    dev = x.device
    z = x * stats["std"].to(dev) + stats["mean"].to(dev)
    return vae.decode(z.transpose(1, 2)).float()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", required=True, help="D2 checkpoint")
    ap.add_argument("--vae", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    g = cfg.grpo
    rank, world, device = setup_distributed()
    seed_everything(7 + rank)
    out_dir = Path(cfg.train.out_dir)
    log = Logger(out_dir, rank == 0)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    ck = load_checkpoint(args.init)
    stats = ck["latent_stats"]
    policy = DubFlow(cfg.model).to(device)
    policy.load_state_dict(ck.get("ema") or ck["model"])
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    step = 0
    opt = torch.optim.AdamW(policy.parameters(), g.lr, betas=(0.9, 0.95), weight_decay=0.0)
    last = out_dir / "last.pt"
    if last.exists():
        ck2 = load_checkpoint(last)
        policy.load_state_dict(ck2["model"])
        opt.load_state_dict(ck2["opt"])
        step = ck2["step"]
    vae = load_vae(args.vae, device)
    check_vae_compat(vae, cfg.model.latent_dim, stats)
    spk = SpeakerEmbedder(g.get("speaker_backend", "wavlm_sv"), str(device), SAMPLE_RATE, g.get("speaker_dim"))
    asr = ASR(g.asr_model, str(device)) if g.get("asr_model") else None

    stats_ds = dict(stats)
    cfg.data.ctx_p = 0.0  # rollouts generate whole scenes (no known-context prefix)
    ds = DubDataset(cfg.data.dub_manifest, cfg.data.dub_src_cache, cfg.data.dub_tgt_cache, stats_ds, cfg.data,
                    cfg.model.get("n_streams", 2))
    rng = random.Random(1234 + rank + step)
    G = int(g.group_size)
    while step < g.max_steps:
        idxs = [rng.randrange(len(ds)) for _ in range(g.prompts_per_step)]
        batch = to_device(collate([ds[i] for i in idxs]), device)
        batch_g = repeat_batch(batch, G)
        start = rng.randint(g.window_start_min, g.window_start_max)
        window = (start, start + g.window)
        policy.eval()
        tr = rollout(policy, batch_g, g.steps, window, g.noise_level, g.guidance)
        out_wav = decode(vae, tr.final, stats)
        src_wav = decode(vae, batch["src_latent"], stats)
        parts = []
        for j in range(out_wav.shape[0]):
            p = j // G
            n = int(batch_g["frame_mask"][j].sum())
            o = out_wav[j, : n * LATENT_HOP].cpu()
            s = src_wav[p, : n * LATENT_HOP].cpu()
            row = ds.rows[idxs[p]]
            chunks = [(int(c["t0"] * FRAME_RATE), int(c["t1"] * FRAME_RATE)) for c in row["chunks"]]
            part = {"sim": island_similarity(spk, s, o), "ssp": ssp(spk, s, o), "boundary": boundary_error(o, chunks)}
            if asr is not None:
                ref_text = " ".join(c["text"] for c in row["chunks"])
                part["cer"] = cer(ref_text, asr(o, row.get("tgt_lang", "fa")))
            parts.append(part)
        rewards = combine_rewards(parts, G, dict(g.reward_weights), dict(g.gates))
        adv = group_advantages(rewards, G).to(device)
        policy.train()
        loss, info = grpo_loss(policy, ref, batch_g, tr, adv, g.noise_level, g.guidance, g.clip, g.kl_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:  # manual gradient averaging (the policy is called through several methods)
            for prm in policy.parameters():
                if prm.grad is None:
                    prm.grad = torch.zeros_like(prm)
                dist.all_reduce(prm.grad)
                prm.grad /= world
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        step += 1
        if step % cfg.train.log_every == 0:
            sims = [p["sim"] for p in parts if p["sim"] is not None]
            log.log(step, {"loss": float(loss.detach()), "reward": float(rewards.mean()), "gnorm": float(gn),
                           "sim": sum(sims) / max(1, len(sims)),
                           "boundary": sum(p["boundary"] for p in parts) / len(parts), **info})
        if rank == 0 and (step % cfg.train.save_every == 0 or step == g.max_steps):
            payload = dict(config=cfg.to_dict(), model=unwrap(policy).state_dict(), ema=None,
                           opt=opt.state_dict(), step=step, latent_stats=stats)
            save_checkpoint(last, **payload)
            payload.pop("opt")
            save_checkpoint(out_dir / f"dubflow_grpo_{step:06d}.pt", **payload)
    cleanup_distributed()


if __name__ == "__main__":
    main()
