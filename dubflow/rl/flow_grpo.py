"""Flow-GRPO for DubFlow (rectified flow, t: 0 = noise -> 1 = data).

The deterministic ODE dx = v dt is replaced, inside a short window of steps
(MixGRPO-style), by the marginal-preserving SDE

    dx = [v - (s^2 / 2) * (x - t v) / (1 - t)] dt + s dW,   s(t) = a * sqrt((1 - t) / t)

(score of the linear interpolant: -(x - t v) / (1 - t)). Each stochastic step is
a Gaussian transition whose log-probability is differentiable w.r.t. the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..model.dubflow import Drop


def guided_velocity(model, x, t, cond, uncond, scale: float):
    v, _ = model.velocity(x, t, cond)
    if uncond is not None and scale:
        vu, _ = model.velocity(x, t, uncond)
        v = v + scale * (v - vu)
    return v


def sde_mean_std(x, v, t: float, dt: float, a: float):
    t_c = max(t, 0.05)
    sigma = a * math.sqrt((1 - t_c) / t_c)
    drift = v - 0.5 * sigma ** 2 * (x - t_c * v) / max(1 - t_c, 1e-3)
    return x + drift * dt, sigma * math.sqrt(dt)


def gaussian_logp(x_next, mean, std, mask):
    lp = -((x_next.float() - mean.float()) ** 2) / (2 * std ** 2)
    m = mask[..., None].float()
    return (lp * m).sum((1, 2)) / (m.sum((1, 2)) * x_next.shape[-1]).clamp_min(1)


@dataclass
class Trajectory:
    xs: list  # x_k for stochastic steps
    xn: list  # x_{k+1}
    ts: list
    dts: list
    logp: list
    final: torch.Tensor


def conds(model, batch: dict, guidance: float):
    b = batch["frame_mask"].shape[0]
    dev = batch["frame_mask"].device
    cond = model.prepare(batch, model.full_keep(b, dev))
    uncond = None
    if guidance:
        z = torch.zeros(b, device=dev)
        uncond = model.prepare(batch, Drop(text=z, timbre=z, tracks={k: z for k in model.track_groups}))
    return cond, uncond


@torch.no_grad()
def rollout(model, batch: dict, steps: int, window: tuple[int, int], a: float, guidance: float, generator=None) -> Trajectory:
    cond, uncond = conds(model, batch, guidance)
    mask = batch["frame_mask"]
    b, t = mask.shape
    x = torch.randn(b, t, model.latent_dim, device=mask.device, generator=generator)
    ts = torch.linspace(0, 1, steps + 1).tolist()
    tr = Trajectory([], [], [], [], [], None)
    for k in range(steps):
        t0, dt = ts[k], ts[k + 1] - ts[k]
        tb = torch.full((b,), t0, device=mask.device)
        v = guided_velocity(model, x, tb, cond, uncond, guidance)
        if window[0] <= k < window[1]:
            mean, std = sde_mean_std(x, v, t0, dt, a)
            nxt = mean + std * torch.randn(mean.shape, device=mean.device, generator=generator)
            nxt = nxt * mask[..., None]
            tr.xs.append(x)
            tr.xn.append(nxt)
            tr.ts.append(t0)
            tr.dts.append(dt)
            tr.logp.append(gaussian_logp(nxt, mean, std, mask))
            x = nxt
        else:
            x = (x + dt * v) * mask[..., None]
    tr.final = x
    return tr


def grpo_loss(policy, ref, batch: dict, tr: Trajectory, adv: torch.Tensor, a: float, guidance: float,
              clip: float, kl_beta: float) -> tuple[torch.Tensor, dict]:
    mask = batch["frame_mask"]
    b = mask.shape[0]
    cond, uncond = conds(policy, batch, guidance)
    with torch.no_grad():
        rcond, runcond = conds(ref, batch, guidance) if kl_beta else (None, None)
    total, kls, ratios = 0.0, [], []
    for x, xn, t0, dt, lp_old in zip(tr.xs, tr.xn, tr.ts, tr.dts, tr.logp):
        tb = torch.full((b,), t0, device=mask.device)
        v = guided_velocity(policy, x, tb, cond, uncond, guidance)
        mean, std = sde_mean_std(x, v, t0, dt, a)
        lp = gaussian_logp(xn, mean, std, mask)
        ratio = torch.exp(lp - lp_old)
        surr = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        loss = -surr.mean()
        if kl_beta:
            with torch.no_grad():
                vr = guided_velocity(ref, x, tb, rcond, runcond, guidance)
                mr, _ = sde_mean_std(x, vr, t0, dt, a)
            m = mask[..., None].float()
            kl = (((mean.float() - mr.float()) ** 2) / (2 * std ** 2) * m).sum() / (m.sum() * x.shape[-1]).clamp_min(1)
            loss = loss + kl_beta * kl
            kls.append(float(kl.detach()))
        total = total + loss / len(tr.xs)
        ratios.append(float(ratio.detach().mean()))
    return total, {"kl": sum(kls) / max(1, len(kls)), "ratio": sum(ratios) / max(1, len(ratios))}


def group_advantages(rewards: torch.Tensor, group: int) -> torch.Tensor:
    r = rewards.view(-1, group)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    return adv.view(-1)


def combine_rewards(parts: list[dict], group: int, weights: dict, gates: dict) -> torch.Tensor:
    """AuK-style constrained fusion: candidates failing a hard gate get the
    lowest rewards; the soft terms are z-scored only among passing candidates
    of the same group (so a good voice cannot compensate wrong content).
    Failing candidates are ranked by how much they violate the gates, so a
    group in which nobody passes still yields a learning signal."""
    n = len(parts)
    out = torch.zeros(n)
    for g0 in range(0, n, group):
        idx = list(range(g0, min(n, g0 + group)))
        passing = []
        for i in idx:
            p = parts[i]
            violation = 0.0
            for key in ("cer", "boundary"):
                if p.get(key) is not None and key in gates and p[key] > gates[key]:
                    violation += (p[key] - gates[key]) / max(gates[key], 1e-6)
            out[i] = -2.0 - min(violation, 10.0)
            if violation == 0.0:
                passing.append(i)
        if not passing:
            continue
        score = torch.zeros(len(passing))
        for key, w in weights.items():
            vals = [parts[i].get(key) for i in passing]
            if any(v is None for v in vals) or w == 0:
                continue
            v = torch.tensor(vals, dtype=torch.float)
            z = (v - v.mean()) / (v.std(unbiased=False) + 1e-4) if len(vals) > 1 else torch.zeros(1)
            if key in ("cer", "boundary"):
                z = -z
            score += w * z
        for j, i in enumerate(passing):
            out[i] = score[j]
    return out
