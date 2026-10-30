"""AuK-VAE backend: vendored network, 25 Hz x 128 adapter (patch-2 folding),
strict weight loading, compatibility checks and a pipeline run through it.

A tiny AuK architecture (same code, small channels; latent 8 -> folded 16) keeps
the tests fast and download-free. Set DUBFLOW_AUK_VAE=/path/to/vae.safetensors to
also test the released weights."""

from __future__ import annotations

import os

import pytest
import torch

from dubflow import LATENT_HOP, SAMPLE_RATE
from dubflow.features.backends import EmotionEmbedder, SpeakerEmbedder
from dubflow.infer import dub as dub_mod
from dubflow.preprocess import build_index, extract
from dubflow.train import train_dubflow
from dubflow.utils import checkpoint as ckpt_mod
from dubflow.utils.checkpoint import check_vae_compat, load_dubflow
from dubflow.vae.auk_vae import AuKVAE, AuKVAEAdapter, is_auk_spec, load_auk_vae
from dubflow.vae.model import build_vae

from .conftest import ROOT, synth_voice

TINY = {"upsample_initial_channel": 64, "downsample_channels": [4, 4, 8, 8, 16, 16, 32], "latent_dim": 8}


def tiny_adapter(seed: int = 0) -> AuKVAEAdapter:
    torch.manual_seed(seed)
    return AuKVAEAdapter(AuKVAE(TINY)).eval()


def test_adapter_shapes_and_folding():
    vae = tiny_adapter()
    assert vae.latent_dim == 16 and vae.hop == LATENT_HOP == 960
    wav = 0.1 * torch.randn(2, int(SAMPLE_RATE * 1.3))
    z, mu, logvar = vae.encode(wav)
    t = -(-wav.shape[1] // 960)
    assert mu.shape == (2, 16, t) and logvar.shape == mu.shape and torch.equal(z, mu)
    with torch.no_grad():
        native, log_std = vae.vae.encode_native(vae.pad_to_hop(wav))
    assert native.shape == (2, 8, 2 * t)
    # frame k of the 25 Hz grid = [native frame 2k | native frame 2k+1]
    assert torch.equal(mu[:, :8, 3], native[:, :, 6]) and torch.equal(mu[:, 8:, 3], native[:, :, 7])
    assert torch.allclose(logvar[:, :8, 3], 2 * log_std[:, :, 6])
    assert torch.equal(vae._unfold(vae._fold(native)), native)
    with torch.no_grad():
        y = vae.decode(mu)
    assert y.shape == (2, t * 960) and y.abs().max() <= 1.0


def test_decoder_is_causal_with_short_lookahead():
    vae = tiny_adapter()
    z = torch.randn(1, 16, 40)
    z2 = z.clone()
    z2[..., 20:] += 1.0  # change the future (25 Hz frames >= 20 = native frames >= 40)
    with torch.no_grad():
        a, b = vae.decode(z), vae.decode(z2)
    # 3-frame look-ahead of the input conv at 50 Hz (+ a few samples from anti-aliasing filters)
    cut = (40 - 4) * 480
    assert torch.allclose(a[:, :cut], b[:, :cut], atol=1e-6)
    assert (a[:, 20 * 960 :] - b[:, 20 * 960 :]).abs().max() > 1e-4


def test_strict_loading_from_safetensors(tmp_path):
    from safetensors.torch import save_file

    src = AuKVAE(TINY)
    state = {k: v.contiguous() for k, v in src.state_dict().items()}
    state["flow.flows.0.pre.weight"] = torch.zeros(3)  # training-only flow weights are ignored
    path = tmp_path / "vae.safetensors"
    save_file(state, str(path))
    assert is_auk_spec(path) and is_auk_spec(tmp_path) and is_auk_spec("auk") and not is_auk_spec(tmp_path / "x.pt")
    vae = load_auk_vae(str(tmp_path), cfg=TINY)
    wav = 0.1 * torch.randn(1, SAMPLE_RATE)
    with torch.no_grad():
        ref = AuKVAEAdapter(src.eval()).decode(AuKVAEAdapter(src).encode(wav)[1])
        assert torch.equal(vae.decode(vae.encode(wav)[1]), ref)
    assert not any(p.requires_grad for p in vae.parameters())
    del state["conv_pre.bias"]
    save_file(state, str(path))
    with pytest.raises(RuntimeError, match="conv_pre.bias"):
        load_auk_vae(str(path), cfg=TINY)


def test_vae_compat_checks():
    auk = tiny_adapter()
    check_vae_compat(auk, 16, {"vae_kind": "auk"})
    with pytest.raises(ValueError, match="latent_dim"):
        check_vae_compat(auk, 64)
    with pytest.raises(ValueError, match="'dubvae'"):
        check_vae_compat(auk, 16, {"vae_kind": "dubvae"})
    from dubflow.utils.config import load_config

    dub = build_vae(load_config(ROOT / "configs/vae_tiny.yaml").model)
    check_vae_compat(dub, 16, {"vae_kind": "dubvae"})
    check_vae_compat(dub, 16, {})  # stats from older caches carry no kind
    for name, dim in (("dubflow_m_auk", 128), ("dubflow_s_auk", 128), ("dubflow_m", 64), ("dubflow_s", 64)):
        assert load_config(ROOT / f"configs/{name}.yaml").model.latent_dim == dim


def test_pipeline_on_auk_latents(synth_corpus, tmp_path, monkeypatch):
    """extract -> index -> D1 (with online overlap re-encoding) -> dub(), all through the AuK adapter."""
    vae = tiny_adapter(1)
    monkeypatch.setattr(extract, "load_vae", lambda *a, **k: vae)
    monkeypatch.setattr(ckpt_mod, "load_vae", lambda *a, **k: vae)
    feats = tmp_path / "feats"
    extract.main(["--manifest", str(synth_corpus["manifest"]), "--vae", "auk", "--out", str(feats), "--device", "cpu",
                  "--speaker-backend", "fallback", "--speaker-dim", "16", "--emotion-backend", "fallback",
                  "--emotion-dim", "8"])
    build_index.main(["--manifest", str(synth_corpus["manifest"]), "--cache", str(feats), "--out", str(feats / "index.pt")])
    stats = torch.load(feats / "latent_stats.pt", weights_only=False)
    assert stats["vae_kind"] == "auk" and stats["mean"].numel() == 16
    d1 = tmp_path / "d1"
    common = [f"data.manifest={synth_corpus['manifest']}", f"data.cache={feats}", f"data.index={feats / 'index.pt'}",
              "data.src_aug_versions=0", "data.overlap_p=1.0", "data.scene_p=1.0", f"train.out_dir={d1}"]
    train_dubflow.main(["--config", str(ROOT / "configs/dubflow_tiny.yaml"), "--vae", "auk", *common])
    ck = torch.load(d1 / "last.pt", weights_only=False)
    assert ck["latent_stats"]["vae_kind"] == "auk"
    model, cfg = load_dubflow(d1 / "last.pt")
    src, _ = synth_voice(120, 9.0, seed=3)
    script = {"lang": "fa", "chunks": [{"t0": 0.2, "t1": 2.0, "text": "salam"}, {"t0": 4.0, "t1": 6.5, "text": "khoobi"}]}
    out = dub_mod.dub(model, vae, ck["latent_stats"], src, script, SpeakerEmbedder("fallback", "cpu", SAMPLE_RATE, 16),
                      EmotionEmbedder("fallback", "cpu", SAMPLE_RATE, 8), "cpu", 2, {}, 5.0, context_sec=1.0)
    assert out.shape == src.shape and torch.isfinite(out).all()


@pytest.mark.skipif(not os.environ.get("DUBFLOW_AUK_VAE"), reason="set DUBFLOW_AUK_VAE to the released vae.safetensors")
def test_released_weights_reconstruct_speech():
    vae = load_auk_vae(os.environ["DUBFLOW_AUK_VAE"])
    assert vae.latent_dim == 128
    wav, _ = synth_voice(120, 4.0, seed=2)
    with torch.no_grad():
        y = vae.decode(vae.encode(wav[None])[1])[0, : wav.numel()]
    err = (y - wav).pow(2).sum() / wav.pow(2).sum()
    assert float(10 * torch.log10(1 / err)) > 10  # ~20 dB measured; generous bound
