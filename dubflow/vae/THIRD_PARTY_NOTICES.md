# Third-party notices for `auk_vae.py`

`dubflow/vae/auk_vae.py` adapts the inference part of the AuK-VAE network
(`src/auk/model/vae/` in https://github.com/Tencent-Hunyuan/AuK) so that the
released `vae.safetensors` (https://huggingface.co/tencent/AuK) can be used.

| Component | Source | License |
|---|---|---|
| AuK-VAE network and weights | Tencent AuK, Copyright (C) 2026 Tencent | MIT |
| BigVGAN generator / AMP blocks | NVIDIA BigVGAN, Copyright (c) 2022 NVIDIA CORPORATION | MIT |
| HiFi-GAN generator structure | https://github.com/jik876/hifi-gan, Copyright (c) 2020 Jungil Kong | MIT |
| Anti-aliased activation (up/down-sampling, low-pass filter) | https://github.com/junjun3518/alias-free-torch | Apache-2.0 |
| Kaiser-windowed sinc filter | julius (https://github.com/adefossez/julius), Copyright (c) 2020 Alexandre Défossez | MIT |
| Snake / SnakeBeta activation | https://github.com/EdwardDixon/snake | MIT |

The full license texts are available in the respective repositories (and in
`incl_licenses/` of the BigVGAN / AuK sources). The Apache-2.0 licensed code was
modified (merged into one module, unused options removed).
