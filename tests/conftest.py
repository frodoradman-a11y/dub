"""Synthetic data fixtures (no downloads): harmonic 'voices' with pauses."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import torch

from dubflow import SAMPLE_RATE
from dubflow.utils.audio import save_audio

ROOT = Path(__file__).resolve().parents[1]


def synth_voice(f0: float, dur: float, seed: int, bright: float = 1.0) -> tuple[torch.Tensor, list[dict]]:
    """A harmonic tone with syllable-like envelopes and pauses; returns (wav, words)."""
    g = torch.Generator().manual_seed(seed)
    rnd = random.Random(seed)
    n = int(dur * SAMPLE_RATE)
    t = torch.arange(n) / SAMPLE_RATE
    vib = 1 + 0.03 * torch.sin(2 * math.pi * 5 * t)
    phase = 2 * math.pi * torch.cumsum(f0 * vib / SAMPLE_RATE * torch.ones(n), 0)
    wav = sum((bright ** k) / k * torch.sin(k * phase) for k in range(1, 8))
    env = torch.zeros(n)
    words, pos = [], 0.15
    letters = "abcdefghijklmnopqrstuvwxyz"
    while pos < dur - 0.4:
        wl = rnd.uniform(0.2, 0.45)
        s, e = int(pos * SAMPLE_RATE), int(min(dur, pos + wl) * SAMPLE_RATE)
        env[s:e] = torch.hann_window(e - s, periodic=False)
        words.append({"w": "".join(rnd.choice(letters) for _ in range(rnd.randint(2, 5))), "s": round(pos, 3), "e": round(pos + wl, 3)})
        pos += wl + (rnd.uniform(0.25, 0.5) if rnd.random() < 0.3 else rnd.uniform(0.02, 0.06))
    wav = 0.3 * wav * env + 0.002 * torch.randn(n, generator=g)
    return wav, words


@pytest.fixture(scope="session")
def synth_corpus(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("corpus")
    (root / "wav").mkdir()
    rows = []
    speakers = {"en_a": (110, "en", 0.6), "en_b": (125, "en", 0.7), "en_c": (210, "en", 0.5),
                "fa_a": (115, "fa", 0.65), "fa_b": (220, "fa", 0.55)}
    fa_words = ["سلام", "دوست", "منِ", "خوب", "کجا", "خانه"]
    for si, (spk, (f0, lang, br)) in enumerate(speakers.items()):
        for u in range(3):
            wav, words = synth_voice(f0 * (1 + 0.05 * u), 3.0 + u, seed=si * 10 + u, bright=br)
            if lang == "fa":
                for w in words:
                    w["w"] = random.Random(len(w["w"])).choice(fa_words)
            p = root / "wav" / f"{spk}_{u}.wav"
            save_audio(str(p), wav, SAMPLE_RATE)
            rows.append({"id": f"{spk}_{u}", "audio": str(p), "lang": lang, "speaker": spk,
                         "text": " ".join(w["w"] for w in words), "words": words})
    man = root / "manifest.jsonl"
    man.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    # A small "film": two actors alternate (src = English, tgt = 'voice-converted' dub).
    src_parts, tgt_parts, chunks, t = [], [], [], 0.0
    for k in range(6):
        f0 = 110 if k % 2 == 0 else 128  # two similar male voices back to back
        ws, _ = synth_voice(f0, 2.0, seed=100 + k, bright=0.6)
        wt, _ = synth_voice(f0 * 1.02, 2.0, seed=200 + k, bright=0.6)
        gap = torch.zeros(int((0.6 if k % 3 else 0.05) * SAMPLE_RATE))
        src_parts += [ws, gap]
        tgt_parts += [wt, gap]
        chunks.append({"t0": round(t + 0.1, 3), "t1": round(t + 1.9, 3), "text": "سلام دوستِ من" if k % 2 else "خوب هستی"})
        t += 2.0 + gap.numel() / SAMPLE_RATE
    film_src, film_tgt = root / "film_src.wav", root / "film_tgt.wav"
    save_audio(str(film_src), torch.cat(src_parts), SAMPLE_RATE)
    save_audio(str(film_tgt), torch.cat(tgt_parts), SAMPLE_RATE)
    script = root / "film_fa.json"
    script.write_text(json.dumps({"lang": "fa", "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    return {"root": root, "manifest": man, "rows": rows, "film_src": film_src, "film_tgt": film_tgt, "script": script}
