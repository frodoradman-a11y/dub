"""GAN discriminators and losses for DubVAE (HiFi-GAN/BigVGAN family)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ..utils.audio import log_mel


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: int = 32):
        super().__init__()
        self.period = period
        chs = [1, channels, channels * 4, channels * 16, channels * 32, channels * 32]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5)
        )
        self.post = weight_norm(nn.Conv2d(chs[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.view(b, 1, -1, self.period)
        feats = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft: int, hop: int, channels: int = 32):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(2, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(x, self.n_fft, self.hop, window=window, return_complex=True)
        h = torch.stack([spec.real, spec.imag], 1).transpose(2, 3)  # [B,2,T,F]
        feats = []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.1)
            feats.append(h)
        h = self.post(h)
        feats.append(h)
        return h.flatten(1), feats


class Discriminators(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), stft_ffts=(512, 1024, 2048), channels: int = 32):
        super().__init__()
        self.discs = nn.ModuleList(
            [PeriodDiscriminator(p, channels) for p in periods]
            + [STFTDiscriminator(n, n // 4, channels) for n in stft_ffts]
        )

    def forward(self, x: torch.Tensor):
        return [d(x) for d in self.discs]


def discriminator_loss(real_outs, fake_outs) -> torch.Tensor:
    loss = 0.0
    for (r, _), (f, _) in zip(real_outs, fake_outs):
        loss = loss + torch.mean((1 - r) ** 2) + torch.mean(f ** 2)
    return loss


def generator_adv_loss(fake_outs) -> torch.Tensor:
    return sum(torch.mean((1 - f) ** 2) for f, _ in fake_outs)


def feature_matching_loss(real_outs, fake_outs) -> torch.Tensor:
    loss = 0.0
    for (_, rf), (_, ff) in zip(real_outs, fake_outs):
        for a, b in zip(rf, ff):
            loss = loss + torch.mean(torch.abs(a.detach() - b))
    return loss


def multiscale_mel_loss(x: torch.Tensor, y: torch.Tensor, sample_rate: int,
                        scales=((32, 8, 5), (64, 16, 10), (128, 32, 20), (256, 64, 40),
                                (512, 128, 80), (1024, 256, 160), (2048, 512, 320))) -> torch.Tensor:
    loss = 0.0
    for n_fft, hop, n_mels in scales:
        mx = log_mel(x, sample_rate, n_fft, hop, n_mels)
        my = log_mel(y, sample_rate, n_fft, hop, n_mels)
        loss = loss + F.l1_loss(mx, my)
    return loss / len(scales)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar)
