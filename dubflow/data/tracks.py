"""Frame-level conditioning tracks and text-to-frame anchoring.

Shared by the training datasets and the inference pipeline so that both build
exactly the same inputs. Frame rate = 25 Hz.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

TRACK_ORDER = ("spk_island", "spk_short", "prosody", "emotion", "activity")
PROSODY_DIM = 3
ACTIVITY_DIM = 3


def track_dims(speaker_dim: int, emotion_dim: int) -> dict[str, int]:
    return {"spk_island": speaker_dim, "spk_short": speaker_dim, "prosody": PROSODY_DIM,
            "emotion": emotion_dim, "activity": ACTIVITY_DIM}


def resample_track(x: torch.Tensor, n_out: int) -> torch.Tensor:
    """Linear resampling of [N, D] -> [n_out, D] (time axis)."""
    if x.shape[0] == n_out:
        return x
    if x.shape[0] == 1:
        return x.expand(n_out, -1).clone()
    return F.interpolate(x.T[None].float(), size=n_out, mode="linear", align_corners=False)[0].T


def island_track(islands: torch.Tensor, spk_island: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, spk_island.shape[-1])
    for (s, e), emb in zip(islands.tolist(), spk_island.float()):
        out[max(0, s) : min(n, e)] = emb
    return out


def smooth_identity(per_frame: torch.Tensor, win: int = 25) -> torch.Tensor:
    """Simulate a ~1 s sliding-window speaker embedding from per-frame
    embeddings (zeros in silence): moving average + renormalization. Around
    speaker changes this blends voices exactly like a real sliding window."""
    if per_frame.shape[0] == 0:
        return per_frame
    x = per_frame.T[None]
    k = torch.ones(x.shape[1], 1, win) / win
    y = F.conv1d(F.pad(x, (win // 2, win - 1 - win // 2), mode="replicate"), k, groups=x.shape[1])[0].T
    norm = y.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, y / norm.clamp_min(1e-6), torch.zeros_like(y))


def prosody_track(f0: torch.Tensor, energy: torch.Tensor, vad: torch.Tensor, smooth: int = 5) -> torch.Tensor:
    """[T, 3]: speaker-normalized log-F0 (0 if unvoiced), level relative to the
    clip peak, voicing. Smoothed with a `smooth`-frame box filter (~5 Hz)."""
    if f0.numel() == 0:
        return torch.zeros(0, 3)
    f0 = f0.float()
    voiced = f0 > 0
    lf0 = torch.zeros_like(f0)
    if voiced.sum() > 2:
        l = torch.log(f0[voiced])
        lf0[voiced] = (l - l.mean()) / (l.std(unbiased=False) + 1e-4)
    en = ((energy.float() - energy.float().max()) / 20.0).clamp(-3, 0) + 1.5
    en = en * vad.float()
    x = torch.stack([lf0, en, voiced.float()], -1)
    if smooth > 1 and x.shape[0] > 1:
        k = torch.ones(3, 1, smooth) / smooth
        x = F.conv1d(F.pad(x.T[None], (smooth // 2, smooth - 1 - smooth // 2), mode="replicate"), k, groups=3)[0].T
    return x


def degrade_prosody(x: torch.Tensor, rng: random.Random, warp: float = 0.15, shift_frames: int = 8,
                    phrase_flatten_p: float = 0.3, islands: list[tuple[int, int]] | None = None) -> torch.Tensor:
    """Make the prosody track coarse and imprecise for monolingual pretraining so
    the model cannot learn to copy the exact contour (it is NOT valid across
    languages; the real mapping is learned from dubbing pairs)."""
    n = x.shape[0]
    if n < 4:
        return x
    factor = 1.0 + rng.uniform(-warp, warp)
    m = max(2, int(round(n * factor)))
    y = resample_track(x, m)
    shift = rng.randint(-shift_frames, shift_frames)
    y = torch.roll(y, shift, 0)
    y = y[:n] if m >= n else torch.cat([y, y[-1:].expand(n - m, -1)], 0)
    if islands:
        for s, e in islands:
            if rng.random() < phrase_flatten_p and e > s:
                y[s:e, 0] = y[s:e, 0].mean()
    return y


def activity_track(vad: torch.Tensor, overlap: torch.Tensor | None = None, nv: torch.Tensor | None = None) -> torch.Tensor:
    n = vad.shape[0]
    ov = overlap.float() if overlap is not None else torch.zeros(n)
    nvt = nv.float() if nv is not None else torch.zeros(n)
    return torch.stack([vad.float(), ov, nvt], -1)


def window_centers_offset(regions: list[tuple[int, int]], n: int, min_offset: int, rng: random.Random) -> torch.Tensor:
    """Offset-window centers for monolingual pretraining: each frame of a
    same-speaker region points into a *different* time of that same region
    (circular shift by >= min_offset), so the timbre window never contains the
    target content. Regions too short get -1 (no window)."""
    c = torch.full((n,), -1, dtype=torch.long)
    for a, b in regions:
        length = b - a
        if length < 2 * min_offset + 1:
            continue
        o = rng.randint(min_offset, length - min_offset)
        idx = torch.arange(a, b)
        c[a:b] = a + (idx - a + o) % length
    return c


@dataclass
class Chunk:
    start: int  # frame, inclusive
    end: int  # frame, exclusive
    text: str
    stream: int = 0
    ok: bool = True


def tokenize_chunks(chunks: list[Chunk], tokenizer, lang: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Concatenate chunk token ids (space-separated) -> (ids, [(tok_start, n_tok)])."""
    ids: list[int] = []
    spans = []
    for i, ch in enumerate(chunks):
        if i:
            ids.append(tokenizer.space_id)
        tok = tokenizer.encode(ch.text, lang) or [tokenizer.space_id]
        spans.append((len(ids), len(tok)))
        ids.extend(tok)
    return ids, spans


def build_frame_tok(chunks: list[Chunk], spans: list[tuple[int, int]], n_frames: int, n_streams: int = 2) -> torch.Tensor:
    """Time-anchored average upsampling index: tokens of each chunk are spread
    uniformly over that chunk's [start, end) frames. Returns [S, T] long."""
    out = torch.full((n_streams, n_frames), -1, dtype=torch.long)
    for ch, (t0, nt) in zip(chunks, spans):
        s, e = max(0, ch.start), min(n_frames, ch.end)
        if e <= s or nt <= 0:
            continue
        f = torch.arange(s, e)
        pos = ((f - s) * nt) // (e - s)
        out[min(ch.stream, n_streams - 1), s:e] = t0 + pos
    return out


def chunks_from_words(words: list[dict], frame_rate: float, offset: int, rng: random.Random,
                      pause_sec: float = 0.2, split_p: float = 0.3, whole_p: float = 0.15) -> list[Chunk]:
    """Random-granularity chunks from word alignments (robust to translator style):
    always split at pauses, sometimes split further at word boundaries, sometimes
    keep the whole utterance as one chunk."""
    if not words:
        return []
    if rng.random() < whole_p:
        groups = [words]
    else:
        groups, cur = [], [words[0]]
        for prev, w in zip(words, words[1:]):
            if w["s"] - prev["e"] >= pause_sec or (rng.random() < split_p and len(cur) >= 2):
                groups.append(cur)
                cur = [w]
            else:
                cur.append(w)
        groups.append(cur)
    out = []
    for g in groups:
        s = offset + int(round(g[0]["s"] * frame_rate))
        e = offset + max(int(round(g[-1]["e"] * frame_rate)), int(round(g[0]["s"] * frame_rate)) + 1)
        out.append(Chunk(s, e, " ".join(w["w"] for w in g)))
    return out
