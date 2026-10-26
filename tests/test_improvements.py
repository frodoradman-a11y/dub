"""Tests for: acoustic matching (EQ + room), source augmentation, scene
continuity (known-context channel) and overlapping speech."""

from __future__ import annotations

import json
import math
import random

import pytest
import torch

from dubflow import FRAME_RATE, LATENT_HOP, SAMPLE_RATE
from dubflow.data.tracks import Chunk, assign_streams, overlap_tracks
from dubflow.features.backends import EmotionEmbedder, SpeakerEmbedder
from dubflow.infer import dub as dub_mod
from dubflow.infer.acoustics import apply_room, decay_stats, fit_room, ltas_distance, match_eq, match_room
from dubflow.model.dubflow import DubFlow
from dubflow.train.distill import teacher_target
from dubflow.train.train_dubflow import reencode_overlaps
from dubflow.utils.audio import save_audio
from dubflow.utils.augment import SourceAugmenter, add_reverb, band_limit, eq_tilt, mix_at_snr
from dubflow.utils.config import load_config
from dubflow.vae.model import build_vae

from .conftest import ROOT, synth_voice
from .test_model import CONFIGS, make_batch

SR = SAMPLE_RATE


# --------------------------------------------------------------------------- acoustics
@pytest.fixture(scope="module")
def voices():
    a, _ = synth_voice(120, 10.0, 5)
    b, _ = synth_voice(118, 10.0, 6)  # e.g. the dub voice-converted to the same actor
    return a, b


@pytest.mark.parametrize("channel", ["bright", "dull", "band_limited"])
def test_eq_matching_reduces_spectral_distance(voices, channel):
    dry, other = voices
    ref = {"bright": lambda x: eq_tilt(x, 3.0, SR), "dull": lambda x: eq_tilt(x, -4.0, SR),
           "band_limited": lambda x: band_limit(x, 5000, SR)}[channel](dry)
    before = ltas_distance(other, ref, SR)
    y = match_eq(other, ref, SR)
    after = ltas_distance(y, ref, SR)
    assert after < (0.25 if channel != "band_limited" else 0.7) * before
    assert torch.allclose(y.pow(2).mean(), other.pow(2).mean(), rtol=1e-3)  # level is left to loudness matching
    assert torch.equal(match_eq(other, torch.zeros_like(other), SR), other)  # silent reference: no-op


def test_decay_stats_monotonic_in_reverb(voices):
    dry, _ = voices
    stats = [decay_stats(add_reverb(dry, rt, wet, SR, seed=1), SR) for rt, wet in ((0.2, 0.2), (0.5, 0.4), (1.0, 0.8))]
    assert all(s is not None for s in stats)
    tails = [s["tail_db"] for s in stats]
    rts = [s["decay_rt"] for s in stats]
    assert tails[0] < tails[1] < tails[2]
    assert rts[0] < rts[1] < rts[2]
    assert decay_stats(dry, SR)["tail_db"] < tails[0] - 3


@pytest.mark.parametrize("rt60,wet", [(0.3, 0.4), (0.6, 0.6), (1.0, 0.8)])
def test_room_matching_reproduces_original_reverb(voices, rt60, wet):
    dry, other = voices
    ref = mix_at_snr(add_reverb(dry, rt60, wet, SR, seed=1), torch.randn(dry.numel()), 35)
    y, info = match_room(other, ref, SR)
    assert info["wet"] > 0
    a, b, c = decay_stats(ref, SR), decay_stats(other, SR), decay_stats(y, SR)
    assert abs(c["tail_db"] - a["tail_db"]) < 3.0 < abs(b["tail_db"] - a["tail_db"])
    assert abs(math.log(c["decay_rt"] / a["decay_rt"])) < math.log(1.35)
    assert y.abs().max() <= 0.99 + 1e-6


def test_room_matching_never_adds_reverb_to_match_a_dry_original(voices):
    dry, other = voices
    _, info = match_room(other, dry, SR)
    assert info["wet"] == 0.0
    _, info = match_room(add_reverb(other, 0.8, 0.7, SR), dry, SR)  # dub already wetter: leave it
    assert info["wet"] == 0.0
    assert fit_room(other, None, SR)["wet"] == 0.0  # unmeasurable original


def test_apply_room_keeps_the_tail(voices):
    dry, _ = voices
    y = apply_room(dry, 0.8, 0.5, SR, tail_sec=1.0)
    assert y.numel() == dry.numel() + SR
    assert y[-SR // 2 :].abs().max() > 0  # reverb continues after the input ends
    assert torch.equal(apply_room(dry, 0.5, 0.0, SR, tail_sec=0.5)[: dry.numel()], dry)


# --------------------------------------------------------------------------- augmentation
def test_source_augmenter(tmp_path, voices):
    dry, _ = voices
    noise = tmp_path / "noise.wav"
    save_audio(str(noise), 0.1 * torch.randn(SR * 3), SR)
    lst = tmp_path / "noise.jsonl"
    lst.write_text(json.dumps({"audio": str(noise)}) + "\n")
    aug = SourceAugmenter(SR, noise_list=str(lst), music_list=str(lst))
    outs = []
    for k in range(12):
        aug.rng = random.Random(f"0:x:{k}")
        y, info = aug(dry)
        assert y.shape == dry.shape and torch.isfinite(y).all() and y.abs().max() <= 0.99 + 1e-6
        outs.append((y, info))
    assert any(info for _, info in outs) and len({tuple(sorted(i)) for _, i in outs}) > 3  # varied chains
    aug.rng = random.Random("0:x:3")
    y3, _ = aug(dry)
    assert torch.equal(y3, outs[3][0])  # reproducible per (seed, item, copy)


# --------------------------------------------------------------------------- overlap
def test_assign_streams():
    keep = [Chunk(0, 10, "a", 0), Chunk(5, 15, "b", 1), Chunk(20, 30, "c", 0)]
    assert [c.stream for c in assign_streams(keep)] == [0, 1, 0]  # valid script streams are kept
    clash = [Chunk(0, 10, "a"), Chunk(5, 15, "b"), Chunk(12, 20, "c"), Chunk(25, 30, "d")]
    assert [c.stream for c in assign_streams(clash)] == [0, 1, 0, 0]
    three = [Chunk(0, 10, "a"), Chunk(2, 12, "b"), Chunk(4, 14, "c")]
    out = assign_streams(three, 2)
    assert [c.stream for c in out] == [0, 1, 0] and all(isinstance(c, Chunk) for c in out)
    assert [c.stream for c in assign_streams([Chunk(0, 5, "x", 3)])] == [1]  # clamped to n_streams


def test_overlap_tracks():
    n, d = 120, 8
    a = torch.nn.functional.normalize(torch.randn(d), dim=0)
    b = torch.nn.functional.normalize(torch.randn(d), dim=0)
    short = torch.zeros(n, d)
    short[0:50] = a
    short[50:60] = torch.nn.functional.normalize(a + b, dim=0)  # blended window in the overlap
    short[60:110] = b
    chunks = assign_streams([Chunk(0, 60, "a"), Chunk(50, 110, "b")])
    spk_ov, flag = overlap_tracks(chunks, short, n)
    assert flag[50:60].all() and not flag[:50].any() and not flag[60:].any()
    assert torch.allclose(spk_ov[50:110], b.expand(60, -1), atol=1e-5)  # clean B, from its solo frames
    assert spk_ov[:50].abs().sum() == 0 and spk_ov[110:].abs().sum() == 0
    no_ov, no_flag = overlap_tracks([Chunk(0, 50, "a"), Chunk(60, 100, "b")], short, n)
    assert no_ov.abs().sum() == 0 and not no_flag.any()


def test_track_order_is_validated():
    cfg = load_config(ROOT / "configs/dubflow_tiny.yaml")
    t = dict(cfg.model.tracks)
    cfg.model.tracks = {k: t[k] for k in reversed(list(t))}
    with pytest.raises(ValueError, match="model.tracks"):
        DubFlow(cfg.model)


def _tiny_vae():
    cfg = load_config(ROOT / "configs/vae_tiny.yaml")
    torch.manual_seed(0)
    return build_vae(cfg.model).eval()


def test_reencode_overlaps():
    vae = _tiny_vae()
    t, d = 30, 16
    batch = {
        "x1": torch.randn(2, t, d), "src_latent": torch.randn(2, t, d),
        "frame_mask": torch.ones(2, t, dtype=torch.bool), "reencode": torch.tensor([True, False]),
        "wav": 0.1 * torch.randn(2, t * LATENT_HOP),
    }
    batch["frame_mask"][0, 25:] = False
    keep_x1, keep_src = batch["x1"].clone(), batch["src_latent"].clone()
    stats = {"mean": torch.full((d,), 0.1), "std": torch.full((d,), 2.0)}
    reencode_overlaps(batch, vae, stats)
    with torch.no_grad():
        z = (vae.encode(batch["wav"][:1])[1].transpose(1, 2)[0, :t] - 0.1) / 2.0
    assert torch.allclose(batch["x1"][0, :25], z[:25], atol=1e-5)
    assert torch.equal(batch["x1"][0, 25:], keep_x1[0, 25:])  # padding untouched
    assert torch.equal(batch["src_latent"][0], batch["x1"][0])
    assert torch.equal(batch["x1"][1], keep_x1[1]) and torch.equal(batch["src_latent"][1], keep_src[1])


# --------------------------------------------------------------------------- continuity
def _ctx_model(cfg_path):
    cfg = load_config(ROOT / cfg_path)
    torch.manual_seed(0)
    m = DubFlow(cfg.model).eval()
    assert m.ctx_enabled and m.ctx_proj.weight.abs().sum() == 0  # zero-init: no effect until trained
    for p in m.parameters():  # make the zero-initialized paths (AdaLN gates, outputs, ctx) non-trivial
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    return cfg, m


@pytest.mark.parametrize("cfg_path", CONFIGS)
def test_context_frames_are_kept_and_condition_the_rest(cfg_path):
    cfg, m = _ctx_model(cfg_path)
    batch = make_batch(cfg)
    ctx = torch.randn_like(batch["x1"])
    batch["ctx"] = ctx
    batch["ctx_mask"] = torch.zeros(2, ctx.shape[1], dtype=torch.bool)
    batch["ctx_mask"][:, :10] = True
    x = m.sample(batch, steps=4, seed=1)
    assert torch.allclose(x[:, :10], ctx[:, :10], atol=1e-6)
    # the context changes what is generated after it (through the ctx channel and attention)
    batch2 = dict(batch, ctx=ctx + 1.0)
    x2 = m.sample(batch2, steps=4, seed=1)
    assert (x2[:, 10:] - x[:, 10:]).abs().max() > 1e-4
    # without any context the model is unchanged
    nb = {k: v for k, v in batch.items() if k not in ("ctx", "ctx_mask")}
    assert "ctx" not in m.prepare(nb, m.full_keep(2, "cpu"))


@pytest.mark.parametrize("cfg_path", CONFIGS)
def test_context_training_signal(cfg_path):
    cfg, m = _ctx_model(cfg_path)
    m.train()
    batch = make_batch(cfg)
    batch["ctx_mask"] = torch.zeros(2, batch["x1"].shape[1], dtype=torch.bool)
    batch["ctx_mask"][:, :12] = True
    loss = m(batch)["loss"]
    loss.backward()
    assert m.ctx_proj.weight.grad is not None and m.ctx_proj.weight.grad.abs().sum() > 0
    # the FM loss ignores the known frames: changing the noise target there changes nothing
    torch.manual_seed(3)
    l1 = m(batch)["fm"]
    b2 = dict(batch, loss_mask=batch["loss_mask"].clone())
    b2["loss_mask"][:, :12] = False
    torch.manual_seed(3)
    l2 = m(b2)["fm"]
    assert torch.allclose(l1, l2)


def test_distill_teacher_target_respects_context():
    cfg, m = _ctx_model(CONFIGS[0])
    batch = make_batch(cfg)
    batch["ctx_mask"] = torch.zeros(2, batch["x1"].shape[1], dtype=torch.bool)
    batch["ctx_mask"][:, :8] = True
    x0 = torch.randn_like(batch["x1"])
    t = torch.full((2,), 0.25)
    dt = torch.full((2,), 0.5)
    x_t = 0.75 * x0 + 0.25 * batch["x1"]
    tgt = teacher_target(m, batch, x_t, t, dt, 2, {"all": 1.0}, x0)
    assert torch.allclose(tgt[:, :8], 0.25 * x0[:, :8] + 0.75 * batch["x1"][:, :8], atol=1e-5)


def test_dub_continues_scenes_from_previous_latent(monkeypatch):
    cfg, m = _ctx_model("configs/dubflow_tiny.yaml")
    vae = _tiny_vae()
    spk = SpeakerEmbedder("fallback", "cpu", SR, 16)
    emo = EmotionEmbedder("fallback", "cpu", SR, 8)
    stats = {"mean": torch.zeros(16), "std": torch.ones(16), "silence": torch.zeros(16), "speaker_dim": 16,
             "emotion_dim": 8}
    parts, chunks, t = [], [], 0.0
    for k in range(6):
        w, _ = synth_voice(110 + 20 * (k % 2), 2.0, seed=k)
        parts += [w, torch.zeros(int(0.6 * SR))]
        chunks.append({"t0": t + 0.1, "t1": t + 1.9, "text": "salam"})
        t += 2.6
    # an interruption: overlapping line, no stream given
    chunks.append({"t0": 1.5, "t1": 2.4, "text": "na"})
    src = torch.cat(parts)
    calls = []
    orig = m.sample

    def spy(batch, **kw):
        x = orig(batch, **kw)
        calls.append((batch, x))
        return x

    monkeypatch.setattr(m, "sample", spy)
    out = dub_mod.dub(m, vae, stats, src, {"lang": "fa", "chunks": chunks}, spk, emo, "cpu", 2, {}, 6.0,
                      context_sec=1.0)
    assert out.shape == src.shape and torch.isfinite(out).all()
    assert len(calls) >= 3
    k = int(FRAME_RATE * 1.0)
    assert not calls[0][0]["ctx_mask"].any()
    for (b_prev, x_prev), (b, x) in zip(calls, calls[1:]):
        assert int(b["ctx_mask"].sum()) == k
        n_prev = int(b_prev["frame_mask"][0].sum())
        assert torch.allclose(b["ctx"][0, :k], x_prev[0, n_prev - k : n_prev], atol=1e-6)
        assert torch.allclose(x[0, :k], x_prev[0, n_prev - k : n_prev], atol=1e-6)
    # the interruption was put on its own text stream with an overlap flag
    first = calls[0][0]
    assert (first["frame_tok"][0, 1] >= 0).any() and first["tracks"][0, :, -2].sum() > 0
    # no context requested -> independent scenes
    calls.clear()
    dub_mod.dub(m, vae, stats, src, {"lang": "fa", "chunks": chunks}, spk, emo, "cpu", 2, {}, 6.0,
                acoustic=False, context_sec=0.0)
    assert all(not b["ctx_mask"].any() for b, _ in calls)
