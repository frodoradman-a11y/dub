"""Acoustic matching of the dub to the original recording (post-process).

1. EQ: long-term average spectrum (LTAS) of the dub's speech is matched to the
   original's in ~1/3-octave bands (bounded, smoothed, loudness-neutral).
2. Room: the original's reverberation is summarized blindly from speech (the
   steepest envelope decays bound the room decay rate; energy right after
   phrase offsets gives the tail level). If the original is more reverberant
   than the dub, a synthetic room tail is added, with RT60 and wet gain chosen
   so the dub's statistics match the original's (analysis-by-synthesis).

Loudness is handled separately per chunk (`scene.match_loudness`).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..utils.augment import add_reverb, convolve, synthetic_rir
from ..utils.audio import runs

_NFFT, _HOP = 2048, 512


def _stft_power(x: torch.Tensor) -> torch.Tensor:
    win = torch.hann_window(_NFFT)
    return torch.stft(x, _NFFT, _HOP, window=win, return_complex=True).abs().pow(2)  # [F, T]


def _speech_frames(power: torch.Tensor, rel_db: float = -35.0, floor_dbfs: float = -60.0) -> torch.Tensor:
    """Frames within `rel_db` of the loudest one and above an absolute floor
    (so a silent / near-silent reference yields no frames instead of noise)."""
    win = torch.hann_window(_NFFT)
    ms = power.sum(0) * 2 / (_NFFT * win.pow(2).sum())  # ~ mean square of the frame (Parseval, one-sided)
    e = 10 * torch.log10(ms.clamp_min(1e-12))
    return (e > (e.max() + rel_db)) & (e > floor_dbfs)


def _band_edges(sample_rate: int, fmin: float = 80.0, per_octave: int = 3) -> np.ndarray:
    n = int(per_octave * math.log2((sample_rate / 2) / fmin))
    return fmin * 2 ** (np.arange(n + 1) / per_octave)


def ltas_bands(x: torch.Tensor, sample_rate: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Mean speech power per 1/3-octave band (dB) and band centers (Hz)."""
    p = _stft_power(x)
    sp = _speech_frames(p)
    if sp.sum() < 5:
        return None
    mean = p[:, sp].mean(1).numpy()
    freqs = np.linspace(0, sample_rate / 2, mean.shape[0])
    edges = _band_edges(sample_rate)
    db, centers = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            db.append(10 * np.log10(mean[m].mean() + 1e-12))
            centers.append(math.sqrt(lo * hi))
    return np.array(db), np.array(centers)


def match_eq(out: torch.Tensor, ref: torch.Tensor | None, sample_rate: int, max_boost_db: float = 9.0,
             max_cut_db: float = 24.0, ref_bands: tuple[np.ndarray, np.ndarray] | None = None) -> torch.Tensor:
    """Smoothed per-band EQ that moves the dub's speech LTAS to the original's
    (`ref_bands` = precomputed ltas_bands of the reference). Cuts may be deep
    (band-limited / dull originals), boosts are limited (no noise blow-up).
    The overall RMS is preserved (level is matched per line elsewhere)."""
    a = ref_bands if ref_bands is not None else (ltas_bands(ref, sample_rate) if ref is not None else None)
    b = ltas_bands(out, sample_rate)
    if a is None or b is None:
        return out
    diff = (a[0] - a[0].mean()) - (b[0] - b[0].mean())
    diff = np.convolve(np.pad(diff, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # smooth across bands
    gain = np.clip(diff, -max_cut_db, max_boost_db)
    freqs = np.linspace(0, sample_rate / 2, _NFFT // 2 + 1)
    g = np.interp(np.log(np.maximum(freqs, 20.0)), np.log(a[1]), gain)
    g_lin = torch.from_numpy(10 ** (g / 20)).float()
    win = torch.hann_window(_NFFT)
    spec = torch.stft(out, _NFFT, _HOP, window=win, return_complex=True)
    y = torch.istft(spec * g_lin[:, None], _NFFT, _HOP, window=win, length=out.numel())
    return y * (out.pow(2).mean().sqrt() / y.pow(2).mean().sqrt().clamp_min(1e-8))


def _envelope_db(x: torch.Tensor, sample_rate: int, frame_ms: float = 10.0) -> tuple[torch.Tensor, int]:
    hop = int(sample_rate * frame_ms / 1000)
    n = x.numel() // hop
    e = x[: n * hop].view(n, hop).pow(2).mean(1)
    return 10 * torch.log10(e.clamp_min(1e-12)), hop


def decay_stats(x: torch.Tensor, sample_rate: int, min_gap_ms: float = 150.0) -> dict | None:
    """Reverberation statistics of a speech recording.

    - "decay_rt": blind decay-rate estimate. Speech cannot decay faster than the
      room lets it, so the steepest energy decays (10th percentile of 150 ms
      envelope slopes starting near speech level) bound the room decay rate;
      reported as -60 / slope (seconds). Biased low as an absolute RT60 (the
      direct sound falls first), but monotonic in RT60 and wet level, which is
      all analysis-by-synthesis matching needs.
    - "tail_db": energy 50-150 ms after phrase offsets (strong run followed by
      >= min_gap ms without strong energy) relative to the 100 ms before them,
      noise floor removed (median over offsets).
    """
    env, _ = _envelope_db(x, sample_rate)
    if env.numel() < 50 or env.max() < -60:  # too short / (near) silent
        return None
    env = torch.nn.functional.avg_pool1d(env[None, None], 3, 1, 1, count_include_pad=False)[0, 0]
    top = env.max()
    floor_p = 10 ** (torch.quantile(env, 0.03) / 10)

    w = env.unfold(0, 15, 1)
    tc = torch.arange(15).float() * 0.01
    tc = tc - tc.mean()
    slope = ((w - w.mean(1, keepdim=True)) * tc).sum(1) / (tc**2).sum()
    ok = (w[:, 0] > top - 25) & (w[:, -1] < w[:, 0] - 8)
    if ok.sum() < 5:
        return None
    decay_rt = float(-60.0 / torch.quantile(slope[ok], 0.1).clamp(max=-1.0))

    strong = env > (top - 20)
    gap = int(min_gap_ms / 10)
    tails = []
    for s, e in runs(strong):
        if e - s < 5 or e < 10 or e + gap > env.numel() or strong[e : e + gap].any():
            continue
        pre_p = (10 ** (env[e - 10 : e] / 10)).mean()
        tail_p = (10 ** (env[e + 5 : e + 15] / 10)).mean() - floor_p
        tails.append(float(10 * torch.log10(tail_p.clamp_min(1e-12) / pre_p)))
    if len(tails) < 2:
        return None
    return {"decay_rt": decay_rt, "tail_db": float(np.median(tails)), "n": len(tails)}


def _stats_error(a: dict, b: dict) -> float:
    return abs(a["tail_db"] - b["tail_db"]) / 2.0 + abs(math.log(a["decay_rt"] / b["decay_rt"])) / 0.1


def fit_room(out: torch.Tensor, ref_stats: dict | None, sample_rate: int, seed: int = 0, min_diff_db: float = 1.5,
             rt60_grid: tuple[float, ...] = (0.15, 0.2, 0.3, 0.4, 0.55, 0.75, 1.0, 1.3, 1.8),
             wet_grid: tuple[float, ...] = (0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8)) -> dict:
    """Choose (rt60, wet) so that the dub + synthetic room tail has the same
    reverberation statistics as the original (analysis-by-synthesis over a grid;
    the same estimator is applied to both sides, so its absolute bias cancels).

    Only ever adds reverberation: wet = 0 when the original is dry /
    unmeasurable or the dub is already as reverberant."""
    b = decay_stats(out, sample_rate)
    info = {"ref": ref_stats, "out": b, "wet": 0.0, "rt60": 0.0}
    a = ref_stats
    if a is None or b is None:
        return info
    if a["tail_db"] - b["tail_db"] < min_diff_db and a["decay_rt"] < b["decay_rt"] * 1.15:
        return info
    best_err = _stats_error(a, b)
    for rt60 in rt60_grid:
        rir = synthetic_rir(rt60, sample_rate, seed)
        rir[0] = 0.0
        tail = convolve(out, rir)  # linear in wet: out + wet * tail
        for wet in wet_grid:
            s = decay_stats(out + wet * tail, sample_rate)
            if s is None:
                continue
            err = _stats_error(a, s)
            if err < best_err:
                best_err = err
                info.update(wet=float(wet), rt60=float(rt60), matched=s)
    return info


def apply_room(x: torch.Tensor, rt60: float, wet: float, sample_rate: int, seed: int = 0, tail_sec: float = 0.0) -> torch.Tensor:
    """x + wet * room tail; `tail_sec` > 0 extends the output so the reverb tail
    after the last sound is kept (for overlap-add across scenes)."""
    if wet <= 0:
        return torch.nn.functional.pad(x, (0, int(tail_sec * sample_rate)))
    return add_reverb(torch.nn.functional.pad(x, (0, int(tail_sec * sample_rate))), rt60, wet, sample_rate, seed)


def match_room(out: torch.Tensor, ref: torch.Tensor, sample_rate: int, seed: int = 0, **kw) -> tuple[torch.Tensor, dict]:
    info = fit_room(out, decay_stats(ref, sample_rate), sample_rate, seed, **kw)
    y = apply_room(out, info["rt60"], info["wet"], sample_rate, seed)
    peak = y.abs().max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y, info


def match_acoustics(out: torch.Tensor, ref: torch.Tensor, sample_rate: int, seed: int = 0) -> tuple[torch.Tensor, dict]:
    y = match_eq(out, ref, sample_rate)
    return match_room(y, ref, sample_rate, seed)


def ltas_distance(a: torch.Tensor, b: torch.Tensor, sample_rate: int) -> float:
    """Mean absolute (level-normalized) LTAS difference in dB (for tests / eval)."""
    x, y = ltas_bands(a, sample_rate), ltas_bands(b, sample_rate)
    d = (x[0] - x[0].mean()) - (y[0] - y[0].mean())
    return float(np.abs(d).mean())


__all__ = ["ltas_bands", "match_eq", "decay_stats", "fit_room", "apply_room", "match_room", "match_acoustics",
           "ltas_distance"]
