"""AuK-VAE (Tencent AuK, MIT) as a drop-in alternative to DubVAE.

AuK-VAE: 24 kHz <-> 64-dim latent at 50 Hz (hop 480), non-causal encoder,
causal BigVGAN-style decoder with a 3-frame look-ahead, trained on ~3M hours of
speech / music / general audio (AuK technical report, arXiv 2609.08936).

DubFlow works at 25 Hz (all frame tracks, text anchoring and the timbre window
are defined on that grid), so `AuKVAEAdapter` folds every two consecutive 50 Hz
frames into one 128-dim frame (patch size 2 in time) and exposes exactly the
DubVAE interface: hop 960, `encode(wav) -> (z, mu, logvar)` [B, 128, T],
`decode(z) -> wav` [B, N]. A linear layer on folded frames is the same as a
patch-2 input / output projection inside the model.

Only the inference parts are included (the normalizing flow is used for VAE
training only). The network code below is adapted from the AuK repository
(github.com/Tencent-Hunyuan/AuK, MIT), which in turn adapts NVIDIA BigVGAN (MIT),
HiFi-GAN (MIT), alias-free-torch (Apache-2.0), julius (MIT) and snake (MIT); see
THIRD_PARTY_NOTICES.md. Parameter / buffer names match the released
`vae.safetensors` so the weights load strictly.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

AUK_HF_REPO = "tencent/AuK"
AUK_CONFIG = {
    "upsample_rates": [5, 4, 3, 2, 2, 2],
    "upsample_kernel_sizes": [10, 8, 6, 4, 4, 4],
    "upsample_initial_channel": 1536,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "downsample_rates": [2, 2, 2, 3, 4, 5],
    "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
    "latent_dim": 64,
    "causal": True,
    "act_causal": True,
}


# ----------------------------------------------------------------------------- layers
def _init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)


class _Conv1d(nn.Conv1d):
    """Conv1d with optional causal (left-only) padding."""

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int = 1, dilation: int = 1,
                 bias: bool = True, causal: bool = False):
        self.causal = causal
        self.left_padding = dilation * (kernel_size - 1) if causal else 0
        padding = 0 if causal else int((kernel_size * dilation - dilation) / 2)
        super().__init__(cin, cout, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


class _ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, causal: bool = False):
        if causal and kernel_size != 2 * stride:
            raise ValueError("causal ConvTranspose1d needs kernel_size == 2 * stride")
        padding = 0 if causal else (kernel_size - stride) // 2
        super().__init__(cin, cout, kernel_size, stride=stride, padding=padding)
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x[:, :, : -self.stride[0]] if self.causal else x


class _Conv1dS(nn.Module):
    """Weight-normalized 'same'-padded conv of the encoder (parameters under .layer)."""

    def __init__(self, cin: int, cout: int, kernel_size: int = 1, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.layer = weight_norm(nn.Conv1d(cin, cout, kernel_size, stride=stride,
                                           padding=dilation * (kernel_size - 1) // 2, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class _ResStack(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, base: int = 2, nums: int = 6):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(ch, ch, kernel_size, dilation=base**i, padding=base**i)),
                nn.LeakyReLU(),
                weight_norm(nn.Conv1d(ch, ch, kernel_size, dilation=1, padding=1)),
            )
            for i in range(nums)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, latent_dim: int, channels: list[int], factors: list[int]):
        super().__init__()
        layers: list[nn.Module] = [_Conv1dS(1, channels[0], 3), nn.LeakyReLU(0.2, True)]
        for (cin, cout), f in zip(zip(channels[:-1], channels[1:]), factors):
            layers += [_Conv1dS(cin, cout, 2 * f, stride=f), _ResStack(cout, 3, 2, 6), nn.LeakyReLU(0.2, True)]
        layers += [_Conv1dS(channels[-1], 2 * latent_dim, 3)]
        self.generator = nn.Sequential(*layers)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.generator(wav)


class _SnakeBeta(nn.Module):
    """x + 1/b * sin^2(a x), log-scale a and b."""

    def __init__(self, ch: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(ch))
        self.beta = nn.Parameter(torch.zeros(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.exp()[None, :, None]
        b = self.beta.exp()[None, :, None]
        return x + (1.0 / (b + 1e-9)) * torch.sin(x * a).pow(2)


def _kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half = kernel_size // 2
    delta_f = 4 * half_width
    a = 2.285 * (half - 1) * math.pi * delta_f + 7.95
    beta = 0.1102 * (a - 8.7) if a > 50.0 else (0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21.0) if a >= 21.0 else 0.0)
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half, half) + 0.5) if even else (torch.arange(kernel_size) - half)
    filt = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (filt / filt.sum()).view(1, 1, kernel_size)


class _LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int = 12, causal: bool = False):
        super().__init__()
        if causal:
            self.pad_left, self.pad_right = kernel_size - 1, 0
        else:
            even = kernel_size % 2 == 0
            self.pad_left, self.pad_right = kernel_size // 2 - int(even), kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", _kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(x.shape[1], -1, -1), stride=self.stride, groups=x.shape[1])


class _UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio, self.kernel_size = ratio, kernel_size
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer("filter", _kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(c, -1, -1), stride=self.ratio, groups=c)
        return x[..., self.pad_left : -self.pad_right]


class _DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12, causal: bool = False):
        super().__init__()
        self.lowpass = _LowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size, causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class _Activation1d(nn.Module):
    """Anti-aliased activation: 2x upsample -> SnakeBeta -> 2x downsample."""

    def __init__(self, ch: int, causal: bool):
        super().__init__()
        self.act = _SnakeBeta(ch)
        self.upsample = _UpSample1d(2, 12)
        self.downsample = _DownSample1d(2, 12, causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class _AMPBlock1(nn.Module):
    def __init__(self, ch: int, kernel_size: int, dilation: list[int], causal: bool, act_causal: bool):
        super().__init__()
        self.convs1 = nn.ModuleList(weight_norm(_Conv1d(ch, ch, kernel_size, 1, dilation=d, causal=causal))
                                    for d in dilation)
        self.convs2 = nn.ModuleList(weight_norm(_Conv1d(ch, ch, kernel_size, 1, dilation=1, causal=causal))
                                    for _ in dilation)
        self.convs1.apply(_init_weights)
        self.convs2.apply(_init_weights)
        self.activations = nn.ModuleList(_Activation1d(ch, act_causal) for _ in range(2 * len(dilation)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = x + c2(a2(c1(a1(x))))
        return x


# ----------------------------------------------------------------------------- model
class AuKVAE(nn.Module):
    """AuK BigVGANFlowVAE without the training-only flow: 24 kHz <-> [B, 64, T50]."""

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        h = dict(AUK_CONFIG, **(cfg or {}))
        self.h = h
        self.native_dim = h["latent_dim"]
        self.native_hop = math.prod(h["downsample_rates"])
        # Present in the release (AuK's own normalization); unused here: DubFlow keeps its own latent stats.
        self.register_buffer("global_mean", torch.zeros(self.native_dim))
        self.register_buffer("global_log_std", torch.ones(self.native_dim))
        self.audio_encoder = _Encoder(self.native_dim, list(h["downsample_channels"]), list(h["downsample_rates"]))
        ch0 = h["upsample_initial_channel"]
        self.conv_pre = weight_norm(_Conv1d(self.native_dim, ch0, 7, 1, causal=False))  # 3-frame look-ahead
        self.ups = nn.ModuleList(
            nn.ModuleList([weight_norm(_ConvTranspose1d(ch0 // 2**i, ch0 // 2 ** (i + 1), k, u, causal=h["causal"]))])
            for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"]))
        )
        self.num_kernels = len(h["resblock_kernel_sizes"])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = ch0 // 2 ** (i + 1)
            for k, d in zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"]):
                self.resblocks.append(_AMPBlock1(ch, k, d, h["causal"], h["act_causal"]))
        self.activation_post = _Activation1d(ch, h["act_causal"])
        self.conv_post = weight_norm(_Conv1d(ch, 1, 7, 1, bias=False, causal=h["causal"]))
        for up in self.ups:
            up.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    @torch.autocast("cuda", enabled=False)
    def encode_native(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav [B, N] (N multiple of the hop) -> (mean, log_std) each [B, 64, N / 480]."""
        stats = self.audio_encoder(wav.float()[:, None])
        return stats.chunk(2, 1)

    @torch.autocast("cuda", enabled=False)
    def decode_native(self, z: torch.Tensor) -> torch.Tensor:
        """[B, 64, T50] -> wav [B, N] in [-1, 1]."""
        x = self.conv_pre(z.float())
        for i, up in enumerate(self.ups):
            for layer in up:
                x = layer(x)
            xs = sum(self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels))
            x = xs / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return x.clamp(-1.0, 1.0)[:, 0]


class AuKVAEAdapter(nn.Module):
    """DubVAE-compatible view of AuK-VAE on the 25 Hz grid (2 frames folded -> 128 dims)."""

    kind = "auk"

    def __init__(self, vae: AuKVAE, fold: int = 2):
        super().__init__()
        self.vae = vae
        self.fold = fold
        self.latent_dim = vae.native_dim * fold
        self.hop = vae.native_hop * fold

    def pad_to_hop(self, wav: torch.Tensor) -> torch.Tensor:
        rem = wav.shape[-1] % self.hop
        return F.pad(wav, (0, self.hop - rem)) if rem else wav

    def _fold(self, x: torch.Tensor) -> torch.Tensor:  # [B, D, T*f] -> [B, f*D, T]
        b, d, t = x.shape
        return x.reshape(b, d, t // self.fold, self.fold).permute(0, 3, 1, 2).reshape(b, self.fold * d, t // self.fold)

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:  # [B, f*D, T] -> [B, D, T*f]
        b, fd, t = x.shape
        d = fd // self.fold
        return x.reshape(b, self.fold, d, t).permute(0, 2, 3, 1).reshape(b, d, t * self.fold)

    def encode(self, wav: torch.Tensor, sample: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """wav [B, N] -> (z, mu, logvar) each [B, 128, ceil(N / 960)]."""
        wav = self.pad_to_hop(wav)
        mean, log_std = self.vae.encode_native(wav)
        mu, logvar = self._fold(mean), self._fold(2 * log_std)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, 128, T] -> wav [B, T * 960]."""
        return self.vae.decode_native(self._unfold(z))


def resolve_auk_weights(spec: str) -> Path:
    """'auk' (download from HuggingFace), a directory with vae.safetensors, or a .safetensors file."""
    if spec.startswith("auk:"):
        spec = spec[4:]
    if spec in ("", "auk"):
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(AUK_HF_REPO, "vae.safetensors"))
    p = Path(spec)
    if p.is_dir():
        p = p / "vae.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"AuK VAE weights not found: {p}")
    return p


def is_auk_spec(spec: str | Path) -> bool:
    s = str(spec)
    return (s == "auk" or s.startswith("auk:") or s.endswith(".safetensors")
            or (Path(s).is_dir() and (Path(s) / "vae.safetensors").exists()))


def load_auk_vae(spec: str, device="cpu", cfg: dict | None = None) -> AuKVAEAdapter:
    from safetensors.torch import load_file

    vae = AuKVAE(cfg)
    state = {k: v for k, v in load_file(str(resolve_auk_weights(spec))).items() if not k.startswith("flow.")}
    vae.load_state_dict(state, strict=True)  # every inference weight must be present
    return AuKVAEAdapter(vae).to(device).eval().requires_grad_(False)
