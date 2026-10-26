"""Stage D4: few-step distillation (teacher with CFG -> step-aware student without CFG).

Simulation-free flow distillation (ZipVoice-Distill / shortcut style): sample
x_t on the data interpolant, let the teacher integrate t -> t + dt with CFG in
`teacher_substeps` Euler steps, and train the student (conditioned on the step
size, DSFlow-style step tokens) to reach the same point in ONE step:

    loss = || x_t + dt * v_student(x_t, t, dt) - x_teacher(t + dt) ||^2 / dt^2

Student may be the same architecture (M -> M few-step) or a smaller one (M -> S).

  torchrun --nproc_per_node=8 -m dubflow.train.distill --teacher runs/dubflow_m_d3/last.pt \
      --config configs/dubflow_s.yaml model.step_cond=true data.stage=dub train.out_dir=runs/dubflow_s_distill
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..data.datasets import collate
from ..model.dubflow import Drop, DubFlow
from ..utils.checkpoint import load_checkpoint, load_dubflow, save_checkpoint, unwrap
from ..utils.config import load_config, save_config
from ..utils.train_utils import (
    EMA,
    DistributedWeightedSampler,
    Logger,
    autocast,
    cleanup_distributed,
    seed_everything,
    setup_distributed,
    to_device,
)
from .train_dubflow import build_dataset


class StudentWrapper(torch.nn.Module):
    """Makes the distillation loss the module forward (for DDP gradient sync)."""

    def __init__(self, student: DubFlow):
        super().__init__()
        self.student = student

    def forward(self, batch, x_t, t, dt, target):
        cond = self.student.prepare(batch, self.student.full_keep(x_t.shape[0], x_t.device))
        v, _ = self.student.velocity(x_t, t, cond, dt)
        pred = x_t + dt[:, None, None] * v
        m = (batch["frame_mask"] & ~batch["ctx_mask"]).float()[..., None]  # known context frames are not predicted
        err = ((pred.float() - target.float()) ** 2 / dt[:, None, None] ** 2 * m).sum()
        return err / (m.sum() * v.shape[-1]).clamp_min(1)


@torch.no_grad()
def teacher_target(teacher: DubFlow, batch: dict, x_t, t, dt, substeps: int, guidance: dict, x0=None):
    b = x_t.shape[0]
    dev = x_t.device
    full = teacher.full_keep(b, dev)
    cond = teacher.prepare(batch, full)
    z = torch.zeros(b, device=dev)
    variants = {}
    for k, s in guidance.items():
        if not s:
            continue
        if k == "all":
            d = Drop(text=z, timbre=z, tracks={n: z for n in teacher.track_groups})
        elif k == "timbre":
            d = Drop(text=full.text, timbre=z, tracks=full.tracks)
        elif k == "text":
            d = Drop(text=z, timbre=full.timbre, tracks=full.tracks)
        else:
            d = Drop(text=full.text, timbre=full.timbre, tracks={n: z for n in teacher.track_groups})
        variants[k] = (s, teacher.prepare(batch, d))
    x = x_t
    h = dt / substeps
    for i in range(substeps):
        ti = t + h * i
        if x0 is not None:
            x = teacher.apply_context(x, x0, ti, cond)
        v, _ = teacher.velocity(x, ti, cond)
        vg = v.clone()
        for s, vc in variants.values():
            vu, _ = teacher.velocity(x, ti, vc)
            vg = vg + s * (v - vu)
        x = x + h[:, None, None] * vg
    if x0 is not None:
        x = teacher.apply_context(x, x0, t + dt, cond)
    return x


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--config", required=True, help="student config (step_cond is forced on)")
    ap.add_argument("--student-init", default=None, help="init student weights (e.g. teacher ckpt for M->M)")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cfg.model.step_cond = True
    dcfg = cfg.get("distill", {})
    rank, world, device = setup_distributed()
    seed_everything(99 + rank)
    out_dir = Path(cfg.train.out_dir)
    log = Logger(out_dir, rank == 0)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, out_dir / "config.yaml")

    teacher, _ = load_dubflow(args.teacher, device)
    teacher.requires_grad_(False)
    stats = load_checkpoint(args.teacher)["latent_stats"]
    student = DubFlow(cfg.model).to(device)
    if args.student_init:
        ck = load_checkpoint(args.student_init)
        student.load_state_dict(ck.get("ema") or ck["model"], strict=False)
    student.backbone.grad_ckpt = bool(cfg.train.get("activation_checkpointing", False))
    wrapper = StudentWrapper(student)
    ddp = DDP(wrapper, device_ids=[device.index] if device.type == "cuda" else None,
              find_unused_parameters=True) if world > 1 else wrapper
    opt = torch.optim.AdamW(student.parameters(), dcfg.get("lr", 2e-5), betas=(0.9, 0.95))
    ema = EMA(student, cfg.train.ema_decay)
    ds = build_dataset(cfg, dict(stats), return_wav=False, seed=rank)
    sampler = DistributedWeightedSampler(getattr(ds, "weights", torch.ones(len(ds))), rank, world)
    dl = iter(DataLoader(ds, cfg.train.batch_size, sampler=sampler, num_workers=cfg.data.num_workers,
                         collate_fn=collate, drop_last=True))
    K = int(dcfg.get("student_steps", 4))
    sub = int(dcfg.get("teacher_substeps", 4))
    guidance = dict(dcfg.get("guidance", cfg.sample.guidance))
    max_steps = int(dcfg.get("max_steps", cfg.train.max_steps))
    rng = random.Random(rank)
    step = 0
    while step < max_steps:
        batch = to_device(next(dl), device)
        x1 = batch["x1"]
        b = x1.shape[0]
        # Student grid: K uniform steps; sometimes train with 1 step (preview mode).
        k_steps = 1 if rng.random() < dcfg.get("one_step_p", 0.2) else K
        k = torch.randint(0, k_steps, (b,), device=device)
        t = k.float() / k_steps
        dt = torch.full((b,), 1.0 / k_steps, device=device)
        x0 = torch.randn_like(x1)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        target = teacher_target(teacher, batch, x_t, t, dt, sub, guidance, x0)
        with autocast(device, cfg.train.precision):
            loss = ddp(batch, x_t, t, dt, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        ema.update(student, step)
        step += 1
        if step % cfg.train.log_every == 0:
            log.log(step, {"distill": float(loss.detach()), "gnorm": float(gn)})
        if rank == 0 and (step % cfg.train.save_every == 0 or step == max_steps):
            payload = dict(config=cfg.to_dict(), model=unwrap(student).state_dict(), ema=ema.state_dict(),
                           step=step, latent_stats=stats, student_steps=K)
            save_checkpoint(out_dir / "last.pt", **payload)
            save_checkpoint(out_dir / f"student_{step:07d}.pt", **payload)
    cleanup_distributed()


if __name__ == "__main__":
    main()
