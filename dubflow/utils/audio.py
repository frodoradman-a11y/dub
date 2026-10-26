"""Audio I/O and spectral helpers that avoid heavy optional dependencies."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    """Load an audio file as mono float32 [T] at `sample_rate`."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T.copy()).mean(0)
    if sr != sample_rate:
        wav = resample(wav, sr, sample_rate)
    return wav


def save_audio(path: str, wav: torch.Tensor, sample_rate: int) -> None:
    wav = wav.detach().float().cpu().clamp(-1, 1).numpy()
    if wav.ndim == 2:
        wav = wav.T
    sf.write(path, wav, sample_rate, subtype="PCM_16")


def resample(wav: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return wav
    import torchaudio.functional as AF

    return AF.resample(wav, orig_sr, new_sr)


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> torch.Tensor:
    """HTK-style triangular mel filterbank [n_mels, n_fft//2+1]."""
    fmax = fmax or sample_rate / 2
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_freqs)
    mels = np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    hz = _mel_to_hz(mels)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-8)
        down = (hi - freqs) / max(hi - ce, 1e-8)
        fb[i] = np.maximum(0, np.minimum(up, down))
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    fb *= enorm[:, None]
    return torch.from_numpy(fb)


def log_mel(wav: torch.Tensor, sample_rate: int, n_fft: int, hop: int, n_mels: int, win: int | None = None) -> torch.Tensor:
    """wav [B, T] -> log-mel [B, n_mels, frames]."""
    win = win or n_fft
    if wav.dim() == 1:
        wav = wav[None]
    window = torch.hann_window(win, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(wav, n_fft, hop, win, window=window, center=True, return_complex=True)
    mag = spec.abs()
    fb = mel_filterbank(sample_rate, n_fft, n_mels).to(wav.device, wav.dtype)
    mel = torch.matmul(fb, mag)
    return torch.log(mel.clamp_min(1e-5))


def frame_energy(wav: torch.Tensor, hop: int, win: int | None = None) -> torch.Tensor:
    """RMS energy in dB per frame: wav [T] -> [frames] with frames = ceil(T / hop)."""
    win = win or hop * 2
    n = math.ceil(wav.shape[-1] / hop)
    pad = win // 2
    x = F.pad(wav[None, None], (pad, pad + n * hop - wav.shape[-1]), mode="reflect" if wav.shape[-1] > pad else "constant")
    frames = x.unfold(-1, win, hop)[0, 0, :n]
    rms = frames.pow(2).mean(-1).clamp_min(1e-10).sqrt()
    return 20 * torch.log10(rms.clamp_min(1e-5))


def estimate_f0(wav: torch.Tensor, sample_rate: int, hop: int, fmin: float = 60.0, fmax: float = 600.0, threshold: float = 0.35) -> tuple[torch.Tensor, torch.Tensor]:
    """Lightweight normalized-autocorrelation F0 estimator.

    Good enough for the *coarse* prosody track (it is smoothed to ~5 Hz later).
    Returns (f0_hz [frames], voiced [frames] bool). Unvoiced frames have f0=0.
    For production you can swap in RMVPE / PENN / pyworld via the feature backend.
    """
    win = int(sample_rate * 0.04)
    n = math.ceil(wav.shape[-1] / hop)
    pad = win // 2
    x = F.pad(wav[None, None], (pad, pad + n * hop), mode="constant")[0, 0]
    frames = x.unfold(-1, win, hop)[:n]
    frames = frames - frames.mean(-1, keepdim=True)
    nfft = 1 << (2 * win - 1).bit_length()
    spec = torch.fft.rfft(frames, nfft)
    ac = torch.fft.irfft(spec.abs() ** 2, nfft)[:, :win]
    ac = ac / ac[:, :1].clamp_min(1e-8)
    lag_min = max(1, int(sample_rate / fmax))
    lag_max = min(win - 1, int(sample_rate / fmin))
    seg = ac[:, lag_min:lag_max]
    peak, idx = seg.max(-1)
    lag = (idx + lag_min).float()
    f0 = sample_rate / lag
    energy = frames.pow(2).mean(-1)
    voiced = (peak > threshold) & (energy > energy.max().clamp_min(1e-10) * 1e-4)
    f0 = torch.where(voiced, f0, torch.zeros_like(f0))
    return f0, voiced


def energy_vad(wav: torch.Tensor, sample_rate: int, hop: int, threshold_db: float = -40.0, min_speech: float = 0.1, min_silence: float = 0.15) -> torch.Tensor:
    """Energy-based VAD relative to the clip's peak level. Returns bool [frames]."""
    e = frame_energy(wav, hop)
    speech = e > (e.max() + threshold_db)
    fr = sample_rate / hop
    speech = _remove_short_runs(speech, value=False, min_len=int(round(min_silence * fr)))
    speech = _remove_short_runs(speech, value=True, min_len=int(round(min_speech * fr)))
    return speech


def _remove_short_runs(mask: torch.Tensor, value: bool, min_len: int) -> torch.Tensor:
    if min_len <= 1:
        return mask
    out = mask.clone()
    n = len(mask)
    i = 0
    while i < n:
        if bool(mask[i]) == value:
            j = i
            while j < n and bool(mask[j]) == value:
                j += 1
            interior = i > 0 and j < n
            if j - i < min_len and interior:
                out[i:j] = not value
            i = j
        else:
            i += 1
    return out


def runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return [start, end) index ranges where mask is True."""
    out = []
    n = len(mask)
    i = 0
    while i < n:
        if bool(mask[i]):
            j = i
            while j < n and bool(mask[j]):
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out
