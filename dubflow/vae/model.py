"""DubVAE: 24 kHz waveform <-> 25 Hz continuous latent.

- Encoder: strided residual conv stack (Snake activations), downsample 960.
- Decoder: BigVGAN-v2-style upsampler with anti-aliasing-free AMP blocks
  (multi-receptive-field fusion, kernels 3/7/11, dilations 1/3/5).
- Optional semantic regularization (RedAE-style): a linear head maps the latent
  mean to a frozen SSL teacher's feature space; trained with MSE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class SnakeBeta(nn.Module):
    """x + 1/beta * sin^2(alpha * x), log-parameterized per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.log_beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp()
        beta = self.log_beta.exp()
        return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x).pow(2)


def _wn_conv(cin: int, cout: int, k: int, stride: int = 1, dilation: int = 1, padding: int | None = None) -> nn.Module:
    if padding is None:
        padding = (k - 1) * dilation // 2
    return weight_norm(nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation, padding=padding))


class ResUnit(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.act1 = SnakeBeta(channels)
        self.conv1 = _wn_conv(channels, channels, kernel, dilation=dilation)
        self.act2 = SnakeBeta(channels)
        self.conv2 = _wn_conv(channels, channels, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class AMPBlock(nn.Module):
    """Multi-receptive-field fusion: average of residual stacks with different kernels."""

    def __init__(self, channels: int, kernels: list[int], dilations: list[int]):
        super().__init__()
        self.stacks = nn.ModuleList(
            nn.Sequential(*[ResUnit(channels, k, d) for d in dilations]) for k in kernels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(s(x) for s in self.stacks) / len(self.stacks)


class Encoder(nn.Module):
    def __init__(self, base_channels: int, strides: list[int], latent_dim: int, res_kernel: int = 7):
        super().__init__()
        self.stem = _wn_conv(1, base_channels, 7)
        blocks = []
        ch = base_channels
        for s in strides:
            nxt = ch * 2
            blocks.append(
                nn.Sequential(
                    ResUnit(ch, res_kernel, 1), ResUnit(ch, res_kernel, 3), ResUnit(ch, res_kernel, 9),
                    SnakeBeta(ch),
                    _wn_conv(ch, nxt, 2 * s, stride=s, padding=(s + 1) // 2),
                )
            )
            ch = nxt
        self.blocks = nn.Sequential(*blocks)
        self.act = SnakeBeta(ch)
        self.proj = _wn_conv(ch, 2 * latent_dim, 3)
        self.out_channels = ch

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(wav[:, None]))
        mu, logvar = self.proj(self.act(h)).chunk(2, dim=1)
        return mu, logvar.clamp(-20, 10)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int, strides: list[int], kernels: list[int], dilations: list[int]):
        super().__init__()
        self.pre = _wn_conv(latent_dim, base_channels, 7)
        ups = []
        ch = base_channels
        for s in strides:
            nxt = max(ch // 2, 16)
            ups.append(
                nn.Sequential(
                    SnakeBeta(ch),
                    weight_norm(nn.ConvTranspose1d(ch, nxt, 2 * s, stride=s, padding=(s + 1) // 2, output_padding=s % 2)),
                    AMPBlock(nxt, kernels, dilations),
                )
            )
            ch = nxt
        self.ups = nn.Sequential(*ups)
        self.act = SnakeBeta(ch)
        self.post = _wn_conv(ch, 1, 7)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.ups(self.pre(z))
        return torch.tanh(self.post(self.act(h)))[:, 0]


class DubVAE(nn.Module):
    kind = "dubvae"

    def __init__(
        self,
        latent_dim: int = 64,
        enc_channels: int = 48,
        dec_channels: int = 1536,
        strides: tuple[int, ...] = (2, 4, 5, 4, 6),
        dec_kernels: tuple[int, ...] = (3, 7, 11),
        dec_dilations: tuple[int, ...] = (1, 3, 5),
        semantic_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hop = math.prod(strides)
        self.encoder = Encoder(enc_channels, list(strides), latent_dim)
        self.decoder = Decoder(latent_dim, dec_channels, list(reversed(strides)), list(dec_kernels), list(dec_dilations))
        self.semantic_head = nn.Linear(latent_dim, semantic_dim) if semantic_dim > 0 else None
        # Latent normalization statistics (filled by `dubflow-extract --compute-stats`
        # or at the end of VAE training) so the flow model sees ~unit-variance inputs.
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

    def pad_to_hop(self, wav: torch.Tensor) -> torch.Tensor:
        rem = wav.shape[-1] % self.hop
        if rem:
            wav = F.pad(wav, (0, self.hop - rem))
        return wav

    def encode(self, wav: torch.Tensor, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """wav [B, T] -> (z, mu, logvar) each [B, D, T/hop]."""
        wav = self.pad_to_hop(wav)
        mu, logvar = self.encoder(wav)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # Normalized latent API used by the flow model: [B, T, D] (time-major).
    @torch.no_grad()
    def encode_normalized(self, wav: torch.Tensor) -> torch.Tensor:
        _, mu, _ = self.encode(wav)
        z = mu.transpose(1, 2)
        return (z - self.latent_mean) / self.latent_std

    @torch.no_grad()
    def decode_normalized(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self.latent_std + self.latent_mean
        return self.decode(z.transpose(1, 2))

    def forward(self, wav: torch.Tensor) -> dict:
        z, mu, logvar = self.encode(wav, sample=self.training)
        recon = self.decode(z)
        out = {"recon": recon, "mu": mu, "logvar": logvar, "z": z}
        if self.semantic_head is not None:
            out["semantic"] = self.semantic_head(mu.transpose(1, 2))
        return out


def build_vae(cfg) -> DubVAE:
    return DubVAE(
        latent_dim=cfg.latent_dim,
        enc_channels=cfg.enc_channels,
        dec_channels=cfg.dec_channels,
        strides=tuple(cfg.strides),
        dec_kernels=tuple(cfg.get("dec_kernels", (3, 7, 11))),
        dec_dilations=tuple(cfg.get("dec_dilations", (1, 3, 5))),
        semantic_dim=cfg.get("semantic_dim", 0),
    )
