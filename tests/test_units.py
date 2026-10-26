import random

import pytest
import torch

from dubflow.data.tracks import (
    Chunk,
    build_frame_tok,
    chunks_from_words,
    smooth_identity,
    tokenize_chunks,
    window_centers_offset,
)
from dubflow.eval.metrics import cer, wer
from dubflow.model.conditioning import gather_windows
from dubflow.rl.flow_grpo import combine_rewards, group_advantages
from dubflow.text.tokenizer import TextTokenizer, normalize_text
from dubflow.utils.config import load_config
from dubflow.vae.model import DubVAE

from .conftest import ROOT


def test_persian_normalization_and_tags():
    tok = TextTokenizer()
    assert normalize_text("كتاب علي ۱۲", "fa") == "کتاب علی 12"
    ids = tok.encode("<emph>سلامِ</emph> دوست <laugh>", "fa")
    assert tok.unk_id not in ids
    assert tok.decode(ids) == "<emph>سلامِ</emph> دوست <laugh>"
    assert tok.stoi["ِ"] in ids  # ezafe kasre is a real token


def test_frame_tok_is_time_anchored():
    tok = TextTokenizer()
    chunks = [Chunk(10, 30, "ab"), Chunk(40, 44, "cdef", stream=1)]
    ids, spans = tokenize_chunks(chunks, tok, "en")
    ft = build_frame_tok(chunks, spans, 50, 2)
    assert (ft[:, :10] == -1).all() and (ft[0, 30:] == -1).all()
    assert ft[0, 10] == spans[0][0] and ft[0, 29] == spans[0][0] + 1
    assert set(ft[1, 40:44].tolist()) == set(range(spans[1][0], spans[1][0] + 4))


def test_offset_window_never_points_to_itself():
    rng = random.Random(0)
    c = window_centers_offset([(0, 100), (120, 140)], 150, min_offset=20, rng=rng)
    idx = torch.arange(100)
    d = (c[:100] - idx).abs()
    assert ((d >= 20) & (d <= 80)).all()
    assert (c[:100] >= 0).all() and (c[:100] < 100).all()  # stays inside its region
    assert (c[120:140] == -1).all() and (c[100:120] == -1).all()  # too short / gap


def test_smooth_identity_blends_only_near_boundary():
    a, b = torch.zeros(8), torch.zeros(8)
    a[0], b[1] = 1, 1
    x = torch.cat([a.expand(50, 8), b.expand(50, 8)])
    y = smooth_identity(x, 25)
    assert torch.allclose(y[10], a) and torch.allclose(y[90], b)
    assert 0 < y[50, 0] < 1 and 0 < y[50, 1] < 1


def test_chunks_from_words_split_at_pauses():
    words = [{"w": "a", "s": 0.0, "e": 0.3}, {"w": "b", "s": 0.32, "e": 0.6}, {"w": "c", "s": 1.2, "e": 1.5}]
    for seed in range(20):
        ch = chunks_from_words(words, 25.0, 0, random.Random(seed), whole_p=0.0)
        assert any(c.text.endswith("c") and c.start == 30 for c in ch)


def test_gather_windows_bounds():
    src = torch.arange(10).float()[None, :, None].expand(1, 10, 2)
    mask = torch.ones(1, 10, dtype=torch.bool)
    centers = torch.tensor([[0, 5, 9, -1]])
    win, valid = gather_windows(src, mask, centers, half=2)
    assert win.shape == (1, 4, 5, 2)
    assert valid[0, 0].tolist() == [False, False, True, True, True]
    assert win[0, 1, :, 0].tolist() == [3, 4, 5, 6, 7]
    assert not valid[0, 3].any()


def test_vae_shapes():
    v = DubVAE(latent_dim=8, enc_channels=4, dec_channels=32)
    w = torch.randn(2, 960 * 7 + 100)
    z, mu, _ = v.encode(w)
    assert mu.shape == (2, 8, 8)
    assert v.decode(z).shape[-1] == 960 * 8


def test_full_size_parameter_budgets():
    from dubflow.model.dubflow import DubFlow, count_parameters

    with torch.device("meta"):
        m = count_parameters(DubFlow(load_config(ROOT / "configs/dubflow_m.yaml").model))
        s = count_parameters(DubFlow(load_config(ROOT / "configs/dubflow_s.yaml").model))
    assert 650 < m["backbone"] < 850 and 700 < m["total"] < 950
    assert 120 < s["backbone"] < 250 and 150 < s["total"] < 300


def test_metrics_and_reward_fusion():
    assert cer("سلام دوست", "سلام دوست") == 0
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)
    parts = [{"sim": 0.9, "ssp": 0.5, "boundary": 0.01}, {"sim": 0.5, "ssp": 0.5, "boundary": 0.01},
             {"sim": 0.99, "ssp": 0.9, "boundary": 0.5}, {"sim": 0.7, "ssp": 0.2, "boundary": 0.0}]
    r = combine_rewards(parts, 4, {"sim": 1.0, "ssp": 1.0}, {"boundary": 0.08})
    assert r[2] == r.min()  # gated out despite the best similarity
    assert r[0] > r[1]
    adv = group_advantages(r, 4)
    assert abs(float(adv.mean())) < 1e-5
    # nobody passes -> still a ranking (smaller violation = higher reward)
    bad = [{"boundary": 0.1}, {"boundary": 0.4}]
    r2 = combine_rewards(bad, 2, {"sim": 1.0}, {"boundary": 0.08})
    assert r2[0] > r2[1] and group_advantages(r2, 2).abs().sum() > 0
