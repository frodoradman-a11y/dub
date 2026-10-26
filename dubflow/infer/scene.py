"""Build model inputs for one scene from the original audio + a time-anchored
target script. This is the exact inference-time counterpart of DubDataset."""

from __future__ import annotations

import torch

from .. import FRAME_RATE, LATENT_HOP, SAMPLE_RATE
from ..data.datasets import collate
from ..data.tracks import (
    Chunk,
    activity_track,
    build_frame_tok,
    island_track,
    prosody_track,
    resample_track,
    tokenize_chunks,
)
from ..preprocess.extract import extract_item
from ..text.tokenizer import LANGS, TextTokenizer
from ..utils.audio import energy_vad, runs


def split_scenes(src: torch.Tensor, chunks: list[dict], max_sec: float, min_sec: float = 3.0) -> list[tuple[int, int]]:
    """Cut the timeline (in frames) at silences that are not inside any chunk."""
    vad = energy_vad(src, SAMPLE_RATE, LATENT_HOP)
    n = len(vad)
    busy = vad.clone()
    for c in chunks:
        busy[int(c["t0"] * FRAME_RATE) : int(c["t1"] * FRAME_RATE) + 1] = True
    max_f, min_f = int(max_sec * FRAME_RATE), int(min_sec * FRAME_RATE)
    scenes, start = [], 0
    while start < n:
        if n - start <= max_f:
            scenes.append((start, n))
            break
        lo, hi = start + min_f, start + max_f
        free = (~busy[lo:hi]).nonzero()
        cut = int(free[-1]) + lo if len(free) else hi
        scenes.append((start, cut))
        start = cut
    return scenes


@torch.no_grad()
def build_scene_batch(src_scene: torch.Tensor, scene_chunks: list[dict], lang: str, vae, spk, emo, stats: dict,
                      n_streams: int, device) -> tuple[dict, dict]:
    """src_scene: original vocals of the scene (24 kHz). scene_chunks: chunks with
    times relative to the scene start (seconds). Returns (batch, raw_features)."""
    feats = extract_item(src_scene, vae, spk, emo, str(device), short_window=True)
    n = int(feats["n_frames"])
    mean, std = stats["mean"].float(), stats["std"].float()
    src_lat = (feats["latent"].float() - mean) / std
    isl = island_track(feats["islands"], feats["spk_island"], n)
    short = resample_track(feats["spk_short"].float(), n)
    pro = prosody_track(feats["f0"], feats["energy"], feats["vad"])
    emo_t = resample_track(feats["emo"].float(), n)
    act = activity_track(feats["vad"])
    chunks = [Chunk(int(round(c["t0"] * FRAME_RATE)), max(int(round(c["t1"] * FRAME_RATE)), int(round(c["t0"] * FRAME_RATE)) + 1),
                    c["text"], int(c.get("stream", 0))) for c in scene_chunks]
    tok = TextTokenizer()
    ids, spans = tokenize_chunks(chunks, tok, lang)
    item = {
        "x1": torch.zeros(n, src_lat.shape[1]),
        "text": torch.tensor(ids or [tok.space_id], dtype=torch.long),
        "frame_tok": build_frame_tok(chunks, spans, n, n_streams),
        "lang": LANGS.get(lang, LANGS["other"]),
        "tracks": torch.cat([isl, short, pro, emo_t, act], -1).float(),
        "src_latent": src_lat,
        "win_center": torch.arange(n),
        "loss_mask": torch.ones(n, dtype=torch.bool),
    }
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
    return batch, feats


def match_loudness(out: torch.Tensor, ref: torch.Tensor, spans: list[tuple[int, int]], lo: float = 0.25, hi: float = 4.0) -> torch.Tensor:
    """Per-chunk RMS matching of the dub to the original (sample spans)."""
    out = out.clone()
    for s, e in spans:
        if e <= s:
            continue
        r = ref[s:e].pow(2).mean().sqrt()
        o = out[s:e].pow(2).mean().sqrt()
        if o > 1e-5 and r > 1e-5:
            out[s:e] *= float((r / o).clamp(lo, hi))
    return out


def speech_runs_seconds(wav: torch.Tensor) -> list[tuple[float, float]]:
    vad = energy_vad(wav, SAMPLE_RATE, LATENT_HOP)
    return [(s / FRAME_RATE, e / FRAME_RATE) for s, e in runs(vad)]
