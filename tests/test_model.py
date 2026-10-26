import copy

import pytest
import torch

from dubflow.model.dubflow import DubFlow
from dubflow.rl.flow_grpo import grpo_loss, rollout
from dubflow.utils.config import load_config

from .conftest import ROOT

CONFIGS = ["configs/dubflow_tiny.yaml", "configs/dubflow_tiny_unet.yaml"]


def make_batch(cfg, b=2, t=40, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = sum(cfg.model.tracks.values())
    d = cfg.model.latent_dim
    batch = {
        "x1": torch.randn(b, t, d, generator=g),
        "frame_mask": torch.ones(b, t, dtype=torch.bool),
        "loss_mask": torch.ones(b, t, dtype=torch.bool),
        "text": torch.randint(5, 60, (b, n), generator=g),
        "text_mask": torch.ones(b, n, dtype=torch.bool),
        "frame_tok": torch.randint(-1, n, (b, 2, t), generator=g),
        "lang": torch.tensor([1] * b),
        "tracks": torch.randn(b, t, c, generator=g),
        "src_latent": torch.randn(b, t, d, generator=g),
        "src_mask": torch.ones(b, t, dtype=torch.bool),
        "win_center": torch.arange(t).expand(b, t).clone(),
    }
    batch["frame_mask"][1, t - 8 :] = False
    batch["win_center"][1, t - 8 :] = -1
    return batch


@pytest.mark.parametrize("cfg_path", CONFIGS)
def test_padding_does_not_leak(cfg_path):
    cfg = load_config(ROOT / cfg_path)
    torch.manual_seed(0)
    m = DubFlow(cfg.model).eval()
    for p in m.parameters():  # make zero-initialized output paths non-trivial
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.02)
    batch = make_batch(cfg)
    x = torch.randn_like(batch["x1"])
    t = torch.full((2,), 0.3)
    cond = m.prepare(batch, m.full_keep(2, "cpu"))
    v1, _ = m.velocity(x, t, cond)
    x2 = x.clone()
    x2[1, -8:] = 100.0  # garbage in padded frames
    v2, _ = m.velocity(x2, t, cond)
    assert torch.allclose(v1[1, :-8], v2[1, :-8], atol=1e-4)
    assert v1[1, -8:].abs().sum() == 0


@pytest.mark.parametrize("cfg_path", CONFIGS)
def test_overfit_single_batch(cfg_path):
    cfg = load_config(ROOT / cfg_path)
    torch.manual_seed(0)
    m = DubFlow(cfg.model)
    m.drop_cfg = {"all": 0.0, "text": 0.0, "timbre": 0.0, "tracks": 0.0}
    batch = make_batch(cfg, b=2, t=24)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = None
    torch.manual_seed(1)
    for i in range(150):
        out = m(batch)
        opt.zero_grad()
        out["fm"].backward()
        opt.step()
        if i == 0:
            first = out["fm"].item()
    with torch.no_grad():
        torch.manual_seed(2)
        last = sum(m(batch)["fm"].item() for _ in range(8)) / 8
    assert last < 0.6 * first


def test_grpo_ratio_is_one_without_update():
    cfg = load_config(ROOT / CONFIGS[0])
    torch.manual_seed(0)
    m = DubFlow(cfg.model).eval()
    ref = copy.deepcopy(m)
    batch = make_batch(cfg)
    tr = rollout(m, batch, steps=6, window=(1, 4), a=0.7, guidance=1.5)
    adv = torch.tensor([1.0, -1.0])
    loss, info = grpo_loss(m, ref, batch, tr, adv, 0.7, 1.5, 0.2, 0.02)
    assert abs(info["ratio"] - 1.0) < 1e-3 and info["kl"] < 1e-8
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_sample_respects_mask_and_seed():
    cfg = load_config(ROOT / CONFIGS[0])
    m = DubFlow(cfg.model).eval()
    batch = make_batch(cfg)
    a = m.sample(batch, steps=3, guidance={"all": 1.5, "timbre": 0.5}, seed=3)
    b = m.sample(batch, steps=3, guidance={"all": 1.5, "timbre": 0.5}, seed=3)
    assert torch.equal(a, b) and a[1, -8:].abs().sum() == 0
