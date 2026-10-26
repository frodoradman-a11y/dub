"""Distributed setup, EMA, LR schedule, logging and samplers shared by trainers.

Launch on N GPUs with:  torchrun --nproc_per_node=N -m dubflow.train.train_dubflow ...
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def setup_distributed() -> tuple[int, int, torch.device]:
    """Returns (rank, world_size, device). Works with and without torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            return rank, world, torch.device("cuda", local)
        return rank, world, torch.device("cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, dev


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for e, p in zip(self.shadow.parameters(), model.parameters()):
            e.lerp_(p.detach().to(e.dtype), 1 - d)
        for e, b in zip(self.shadow.buffers(), model.buffers()):
            e.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()


def warmup_lambda(warmup: int, max_steps: int, decay: str = "constant"):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        if decay == "cosine":
            p = (step - warmup) / max(1, max_steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        return 1.0

    return fn


class Logger:
    def __init__(self, out_dir: str | Path, enabled: bool):
        self.enabled = enabled
        self.tb = None
        self.t0 = time.time()
        if enabled:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            self.file = open(Path(out_dir) / "train_log.txt", "a", encoding="utf-8")
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(Path(out_dir) / "tb"))
            except Exception:  # tensorboard is optional
                self.tb = None

    def log(self, step: int, scalars: dict) -> None:
        if not self.enabled:
            return
        msg = f"step {step} | " + " | ".join(f"{k} {v:.4g}" for k, v in scalars.items()) + f" | {time.time() - self.t0:.0f}s"
        print(msg, flush=True)
        self.file.write(msg + "\n")
        self.file.flush()
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)


class DistributedWeightedSampler(Sampler):
    """Infinite weighted sampling, sharded per rank (e.g. language upsampling)."""

    def __init__(self, weights: torch.Tensor, rank: int, world: int, seed: int = 0):
        self.weights = weights.double() / weights.sum()
        self.rank, self.world, self.seed = rank, world, seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + 1000003 * self.epoch)
        while True:
            idx = torch.multinomial(self.weights, 4096 * self.world, replacement=True, generator=g)
            yield from idx[self.rank :: self.world].tolist()
            self.epoch += 1

    def __len__(self) -> int:
        return 2**62


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if dist.is_available() and dist.is_initialized():
        x = x.clone()
        dist.all_reduce(x)
        x /= dist.get_world_size()
    return x
