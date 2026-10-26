"""Pluggable feature extractors.

Production backends rely on optional packages (`pip install dubflow[features]`):
  speaker : "wavlm_sv"  -> microsoft/wavlm-base-plus-sv (transformers, 512-d)
            "ecapa"     -> speechbrain/spkrec-ecapa-voxceleb (192-d)
  emotion : "emotion2vec" -> emotion2vec_plus_base via funasr (768-d, frame level)
  teacher : "w2v_bert"  -> facebook/w2v-bert-2.0 hidden layer (1024-d) for REPA
"fallback" backends are deterministic, dependency-free signal statistics. They
are meant for unit tests / smoke runs only, NOT for training a real model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.audio import log_mel, resample


def _fixed_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(in_dim, out_dim, generator=g)
    q, _ = torch.linalg.qr(m) if in_dim >= out_dim else (m / in_dim ** 0.5, None)
    return q[:, :out_dim]


class SpeakerEmbedder:
    def __init__(self, backend: str = "wavlm_sv", device: str = "cpu", sample_rate: int = 24000, dim: int | None = None):
        self.backend, self.device, self.sr = backend, device, sample_rate
        if backend == "wavlm_sv":
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
            self.dim = 512
        elif backend == "ecapa":
            from speechbrain.inference.speaker import EncoderClassifier

            self.model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
            self.dim = 192
        elif backend == "fallback":
            self.dim = dim or 64
            self.proj = _fixed_projection(160, self.dim, seed=1234)
        else:
            raise ValueError(f"unknown speaker backend {backend}")

    @torch.no_grad()
    def embed_batch(self, segments: list[torch.Tensor]) -> torch.Tensor:
        """List of 1-D waveforms at self.sr -> [N, dim] L2-normalized."""
        if not segments:
            return torch.zeros(0, self.dim)
        if self.backend == "fallback":
            outs = []
            for s in segments:
                if s.numel() < 400:
                    s = F.pad(s, (0, 400 - s.numel()))
                m = log_mel(s, self.sr, 1024, 240, 80)[0]
                feat = torch.cat([m.mean(-1), m.std(-1, unbiased=False)])
                outs.append(feat @ self.proj)
            return F.normalize(torch.stack(outs), dim=-1)
        wavs16 = [resample(s.float(), self.sr, 16000).cpu().numpy() for s in segments]
        if self.backend == "wavlm_sv":
            inp = self.fe(wavs16, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            emb = self.model(**inp).embeddings
            return F.normalize(emb.float().cpu(), dim=-1)
        lens = torch.tensor([len(w) for w in wavs16], dtype=torch.float)
        batch = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(w) for w in wavs16], batch_first=True)
        emb = self.model.encode_batch(batch.to(self.device), lens / lens.max())[:, 0]
        return F.normalize(emb.float().cpu(), dim=-1)

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        return self.embed_batch([wav])[0]

    def embed_windows(self, wav: torch.Tensor, win_sec: float = 1.0, hop_sec: float = 0.2, batch: int = 64) -> torch.Tensor:
        """Sliding-window track: [ceil(len/hop), dim] centered windows."""
        win, hop = int(win_sec * self.sr), int(hop_sec * self.sr)
        n = max(1, -(-wav.numel() // hop))
        padded = F.pad(wav, (win // 2, win // 2))
        segs = [padded[i * hop : i * hop + win] for i in range(n)]
        outs = [self.embed_batch(segs[i : i + batch]) for i in range(0, n, batch)]
        return torch.cat(outs, 0)


class EmotionEmbedder:
    """Frame-level emotion / voice-quality features, projected to `dim` and
    pooled to `rate` Hz."""

    def __init__(self, backend: str = "emotion2vec", device: str = "cpu", sample_rate: int = 24000, dim: int = 64, rate: float = 5.0):
        self.backend, self.device, self.sr, self.dim, self.rate = backend, device, sample_rate, dim, rate
        if backend == "emotion2vec":
            from funasr import AutoModel

            self.model = AutoModel(model="iic/emotion2vec_plus_base", device=device, disable_update=True)
            self.proj = _fixed_projection(768, dim, seed=4321)
        elif backend == "fallback":
            self.proj = _fixed_projection(40, dim, seed=4321)
        elif backend != "none":
            raise ValueError(f"unknown emotion backend {backend}")

    @torch.no_grad()
    def track(self, wav: torch.Tensor) -> torch.Tensor:
        n = max(1, int(round(wav.numel() / self.sr * self.rate)))
        if self.backend == "none":
            return torch.zeros(n, self.dim)
        if self.backend == "fallback":
            hop = int(self.sr / self.rate)
            m = log_mel(wav, self.sr, 2048, hop, 40)[0].T[:n]
            m = (m - m.mean(0)) / (m.std(0, unbiased=False) + 1e-5)
            feats = m @ self.proj
        else:
            w16 = resample(wav.float(), self.sr, 16000).numpy()
            res = self.model.generate(w16, granularity="frame", extract_embedding=True)
            frames = torch.as_tensor(res[0]["feats"]).float()  # [~50Hz, 768]
            pooled = F.adaptive_avg_pool1d(frames.T[None], n)[0].T
            feats = pooled @ self.proj
        if feats.shape[0] < n:
            feats = F.pad(feats, (0, 0, 0, n - feats.shape[0]))
        return feats[:n]


class SSLTeacher:
    """REPA target features at the latent frame rate (computed online on GPU)."""

    def __init__(self, backend: str = "w2v_bert", device: str = "cpu", sample_rate: int = 24000, layer: int = 16, frame_rate: float = 25.0):
        self.backend, self.device, self.sr, self.layer, self.fr = backend, device, sample_rate, layer, frame_rate
        if backend == "w2v_bert":
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

            self.fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
            self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(device).eval()
            self.dim = self.model.config.hidden_size
        elif backend == "fallback":
            self.dim = 80
        else:
            raise ValueError(f"unknown teacher backend {backend}")

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor, n_frames: int) -> torch.Tensor:
        """wav [B, T] at self.sr -> [B, n_frames, dim]."""
        if self.backend == "fallback":
            hop = int(self.sr / self.fr)
            m = log_mel(wav.float(), self.sr, 1024, hop, 80).transpose(1, 2)
        else:
            w16 = [resample(x.float().cpu(), self.sr, 16000).numpy() for x in wav]
            inp = self.fe(w16, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
            out = self.model(**inp, output_hidden_states=True)
            m = out.hidden_states[self.layer].float()
        m = F.interpolate(m.transpose(1, 2), size=n_frames, mode="linear", align_corners=False).transpose(1, 2)
        return m.to(wav.device)
