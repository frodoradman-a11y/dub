"""End-to-end smoke test of every stage on synthetic data with tiny configs:
D0 VAE -> extract -> index -> D1 pretrain -> dub pairs -> D2 -> D3 GRPO -> D4 distill -> dub CLI.
Uses the dependency-free 'fallback' feature backends (no downloads)."""

from __future__ import annotations

import json

import soundfile as sf
import torch

from dubflow.data.datasets import DubDataset, PretrainDataset, collate
from dubflow.infer import dub as dub_cli
from dubflow.preprocess import build_dub_pairs, build_index, extract
from dubflow.train import distill, train_dubflow, train_grpo, train_vae
from dubflow.utils.checkpoint import load_checkpoint
from dubflow.utils.config import load_config

from .conftest import ROOT

FEAT = ["--speaker-backend", "fallback", "--speaker-dim", "16", "--emotion-backend", "fallback", "--emotion-dim", "8"]


def test_full_pipeline(synth_corpus, tmp_path):
    # ---- D0: VAE
    vae_dir = tmp_path / "vae"
    train_vae.main(["--config", str(ROOT / "configs/vae_tiny.yaml"), f"data.manifest={synth_corpus['manifest']}",
                    f"train.out_dir={vae_dir}"])
    vae_ck = vae_dir / "vae_0000004.pt"
    assert vae_ck.exists()

    # ---- extract + index (pretraining corpus)
    feats = tmp_path / "feats"
    extract.main(["--manifest", str(synth_corpus["manifest"]), "--vae", str(vae_ck), "--out", str(feats),
                  "--device", "cpu", *FEAT])
    build_index.main(["--manifest", str(synth_corpus["manifest"]), "--cache", str(feats), "--out", str(feats / "index.pt")])
    item = torch.load(feats / "en_a_0.pt", weights_only=False)
    assert item["latent"].shape[1] == 16 and item["spk_island"].shape[1] == 16 and item["emo"].shape[1] == 8

    common = [f"data.manifest={synth_corpus['manifest']}", f"data.cache={feats}", f"data.index={feats / 'index.pt'}"]
    # dataset sanity: synthetic scenes compose and collate
    cfg = load_config(ROOT / "configs/dubflow_tiny.yaml", common + ["data.scene_p=1.0"])
    stats = torch.load(feats / "latent_stats.pt", weights_only=False)
    ds = PretrainDataset(cfg.data.manifest, cfg.data.cache, cfg.data.index, stats, cfg.data, return_wav=True)
    b = collate([ds[i] for i in range(3)])
    assert b["wav"].shape[1] == b["x1"].shape[1] * 960
    assert (b["win_center"] != torch.arange(b["x1"].shape[1])).all()  # offset-window: never the same frame
    assert b["tracks"].shape[-1] == 16 + 16 + 3 + 8 + 3

    # ---- D1: pretraining (DiT tiny) and S-style UNet tiny
    d1 = tmp_path / "d1"
    train_dubflow.main(["--config", str(ROOT / "configs/dubflow_tiny.yaml"), "--vae", str(vae_ck), *common,
                        f"train.out_dir={d1}"])
    assert (d1 / "last.pt").exists() and any((d1 / "samples").glob("*.wav"))
    d1u = tmp_path / "d1_unet"
    train_dubflow.main(["--config", str(ROOT / "configs/dubflow_tiny_unet.yaml"), *common, f"train.out_dir={d1u}"])
    assert (d1u / "last.pt").exists()

    # ---- dub pairs from the synthetic 'film'
    dubdir = tmp_path / "dub"
    build_dub_pairs.main(["--film", "f1", "--src-audio", str(synth_corpus["film_src"]), "--tgt-audio",
                          str(synth_corpus["film_tgt"]), "--tgt-script", str(synth_corpus["script"]),
                          "--out", str(dubdir), "--max-sec", "8", "--min-sec", "2"])
    pairs = [json.loads(x) for x in (dubdir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pairs) >= 2
    extract.main(["--manifest", str(dubdir / "src_manifest.jsonl"), "--vae", str(vae_ck), "--out",
                  str(tmp_path / "fsrc"), "--device", "cpu", "--short-window", *FEAT])
    extract.main(["--manifest", str(dubdir / "tgt_manifest.jsonl"), "--vae", str(vae_ck), "--out",
                  str(tmp_path / "ftgt"), "--device", "cpu", *FEAT])
    dub_args = ["data.stage=dub", f"data.dub_manifest={dubdir / 'pairs.jsonl'}", f"data.dub_src_cache={tmp_path / 'fsrc'}",
                f"data.dub_tgt_cache={tmp_path / 'ftgt'}"]
    ds2 = DubDataset(str(dubdir / "pairs.jsonl"), str(tmp_path / "fsrc"), str(tmp_path / "ftgt"),
                     torch.load(tmp_path / "ftgt" / "latent_stats.pt", weights_only=False), cfg.data)
    it = ds2[0]
    assert (it["win_center"] == torch.arange(it["x1"].shape[0])).all()  # synchronous window at dub time
    assert (it["frame_tok"][0] >= 0).any()

    # ---- D2: dubbing fine-tune from D1
    d2 = tmp_path / "d2"
    train_dubflow.main(["--config", str(ROOT / "configs/dubflow_tiny.yaml"), *common, *dub_args,
                        f"train.init_from={d1 / 'last.pt'}", f"train.out_dir={d2}"])
    ck = load_checkpoint(d2 / "last.pt")
    assert "latent_stats" in ck and ck["step"] == 3
    # latent normalization must be inherited from D1, not recomputed from the dub cache
    ck1 = load_checkpoint(d1 / "last.pt")
    assert torch.equal(ck["latent_stats"]["mean"], ck1["latent_stats"]["mean"])

    # ---- D3: Flow-GRPO
    d3 = tmp_path / "d3"
    train_grpo.main(["--config", str(ROOT / "configs/dubflow_tiny.yaml"), "--init", str(d2 / "last.pt"),
                     "--vae", str(vae_ck), *dub_args, f"train.out_dir={d3}"])
    assert (d3 / "last.pt").exists()

    # ---- D4: distillation to a step-conditioned student (M->S style: DiT teacher -> UNet student)
    d4 = tmp_path / "d4"
    distill.main(["--teacher", str(d3 / "last.pt"), "--config", str(ROOT / "configs/dubflow_tiny_unet.yaml"),
                  *dub_args, f"train.out_dir={d4}"])
    assert (d4 / "last.pt").exists()

    # ---- inference CLI with teacher and with the distilled student
    for model_ck, name in ((d3 / "last.pt", "teacher"), (d4 / "last.pt", "student")):
        out = tmp_path / f"dub_{name}.wav"
        dub_cli.main(["--model", str(model_ck), "--vae", str(vae_ck), "--src", str(synth_corpus["film_src"]),
                      "--script", str(synth_corpus["script"]), "--out", str(out), "--bg", str(synth_corpus["film_tgt"]),
                      "--speaker-backend", "fallback", "--emotion-backend", "fallback", "--device", "cpu",
                      "--max-scene-sec", "8"])
        data, sr = sf.read(str(out))
        assert sr == 24000 and len(data) > 24000 * 10 and abs(data).max() <= 1.0
