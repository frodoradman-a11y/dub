"""Audio degradations that mimic real dubbing sources (separated film vocals):
background noise, music / effects bleed, room reverb, band limiting / EQ
tilt, clipping and source-separation artifacts ("musical noise").

Also hosts the synthetic room-impulse-response generator shared with the
acoustic matching post-process (`dubflow.infer.acoustics`).
"""

from __future__ import annotations

import json
import math
import random

import torch
import torch.nn.functional as F

from .audio import load_audio


def synthetic_rir(rt60: float, sample_rate: int, seed: int = 0, length_sec: float | None = None,
                  hf_damping: float = 2.0, predelay_ms: float = 2.0) -> torch.Tensor:
    """Exponentially decaying noise RIR with frequency-dependent decay
    (high frequencies decay `hf_damping` times faster). Unit direct path at 0."""
    rt60 = max(rt60, 0.05)
    n = int((length_sec or min(2.0, 1.5 * rt60)) * sample_rate)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(n, generator=g)
    t = torch.arange(n) / sample_rate
    # 60 dB decay in rt60 seconds -> amplitude exp(-6.91 t / rt60)
    lo = noise * torch.exp(-6.91 * t / rt60)
    hi = noise * torch.exp(-6.91 * hf_damping * t / rt60)
    k = 31
    lp = torch.ones(1, 1, k) / k
    lo = F.conv1d(F.pad(lo[None, None], (k // 2, k // 2)), lp)[0, 0]
    tail = lo + (hi - F.conv1d(F.pad(hi[None, None], (k // 2, k // 2)), lp)[0, 0])
    pre = int(predelay_ms / 1000 * sample_rate)
    tail[:pre] = 0
    tail = tail / tail.norm().clamp_min(1e-8)
    rir = tail.clone()
    rir[0] = 1.0
    return rir


def convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Linear convolution via FFT, output trimmed to len(x)."""
    n = x.numel() + h.numel() - 1
    nfft = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)
    return y[: x.numel()]


def add_reverb(x: torch.Tensor, rt60: float, wet: float, sample_rate: int, seed: int = 0) -> torch.Tensor:
    """Dry + wet * (room tail), keeping the direct path untouched."""
    rir = synthetic_rir(rt60, sample_rate, seed)
    tail = rir.clone()
    tail[0] = 0.0
    return x + wet * convolve(x, tail)


def mix_at_snr(x: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    if noise.numel() < x.numel():
        reps = math.ceil(x.numel() / max(1, noise.numel()))
        noise = noise.repeat(reps)
    noise = noise[: x.numel()]
    px = x.pow(2).mean().clamp_min(1e-10)
    pn = noise.pow(2).mean().clamp_min(1e-10)
    return x + noise * torch.sqrt(px / (pn * 10 ** (snr_db / 10)))


def colored_noise(n: int, exponent: float, seed: int) -> torch.Tensor:
    """1/f^exponent noise (0 = white, 1 = pink, 2 = brown)."""
    g = torch.Generator().manual_seed(seed)
    spec = torch.fft.rfft(torch.randn(n, generator=g))
    f = torch.arange(spec.numel()).float().clamp_min(1.0)
    y = torch.fft.irfft(spec / f ** (exponent / 2), n)
    return y / y.std().clamp_min(1e-8)


def eq_tilt(x: torch.Tensor, db_per_octave: float, sample_rate: int) -> torch.Tensor:
    spec = torch.fft.rfft(x)
    f = torch.linspace(0, sample_rate / 2, spec.numel()).clamp_min(100.0)
    gain = 10 ** (db_per_octave * torch.log2(f / 1000.0) / 20)
    return torch.fft.irfft(spec * gain, x.numel())


def band_limit(x: torch.Tensor, cutoff_hz: float, sample_rate: int) -> torch.Tensor:
    spec = torch.fft.rfft(x)
    f = torch.linspace(0, sample_rate / 2, spec.numel())
    mask = torch.sigmoid((cutoff_hz - f) / 150.0)
    return torch.fft.irfft(spec * mask, x.numel())


def separation_artifacts(x: torch.Tensor, strength: float, seed: int, n_fft: int = 1024) -> torch.Tensor:
    """Random time-frequency holes + residual leakage, like an imperfect source separator."""
    g = torch.Generator().manual_seed(seed)
    win = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, n_fft // 4, window=win, return_complex=True)
    keep = (torch.rand(spec.shape, generator=g) > strength * 0.35).float()
    keep = F.avg_pool2d(keep[None, None], (3, 3), 1, 1)[0, 0]
    out = torch.istft(spec * keep, n_fft, n_fft // 4, window=win, length=x.numel())
    return out


class SourceAugmenter:
    """Random chain of degradations. `noise_list` / `music_list` are optional
    JSONL manifests ({"audio": path}) of real noise / music-and-effects stems;
    without them, coloured noise is used for both."""

    def __init__(self, sample_rate: int, noise_list: str | None = None, music_list: str | None = None, seed: int = 0):
        self.sr = sample_rate
        self.rng = random.Random(seed)
        self.noise = self._read(noise_list)
        self.music = self._read(music_list)

    @staticmethod
    def _read(path: str | None) -> list[str]:
        if not path:
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line)["audio"] for line in f if line.strip()]

    def _clip(self, paths: list[str], n: int, exponent: float) -> torch.Tensor:
        if paths:
            w = load_audio(self.rng.choice(paths), self.sr)
            if w.numel() > n:
                s = self.rng.randint(0, w.numel() - n)
                w = w[s : s + n]
            return w
        return colored_noise(n, exponent, self.rng.randint(0, 10**9))

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        r = self.rng
        info: dict = {}
        y = x.clone()
        if r.random() < 0.5:
            info["rt60"], info["wet"] = r.uniform(0.15, 0.9), r.uniform(0.1, 0.6)
            y = add_reverb(y, info["rt60"], info["wet"], self.sr, seed=r.randint(0, 10**9))
        if r.random() < 0.6:
            info["music_snr"] = r.uniform(8, 30)
            y = mix_at_snr(y, self._clip(self.music, y.numel(), 1.0), info["music_snr"])
        if r.random() < 0.5:
            info["noise_snr"] = r.uniform(10, 35)
            y = mix_at_snr(y, self._clip(self.noise, y.numel(), r.choice([0.0, 1.0, 2.0])), info["noise_snr"])
        if r.random() < 0.4:
            info["sep_artifacts"] = r.uniform(0.2, 1.0)
            y = separation_artifacts(y, info["sep_artifacts"], r.randint(0, 10**9))
        if r.random() < 0.3:
            info["tilt"] = r.uniform(-3, 3)
            y = eq_tilt(y, info["tilt"], self.sr)
        if r.random() < 0.2:
            info["cutoff"] = r.uniform(3500, 9000)
            y = band_limit(y, info["cutoff"], self.sr)
        if r.random() < 0.1:
            info["clip"] = r.uniform(0.3, 0.8)
            peak = y.abs().max().clamp_min(1e-6)
            y = y.clamp(-info["clip"] * peak, info["clip"] * peak)
        peak = y.abs().max()
        if peak > 0.99:
            y = y * (0.99 / peak)
        return y, info
