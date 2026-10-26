"""Evaluation metrics (also used as RL rewards).

- speaker similarity per speech island (output vs. original at the same time)
- SSP: Speaker-Structure Preservation = correlation between the pairwise
  identity-similarity matrices of the original and the dub (label-free)
- boundary error: share of output speech that falls outside the script chunks
- CER / WER via edit distance (ASR backend optional)
- SLC@p (speech length compliance) and pause IoU (X-Dubbing metrics)
"""

from __future__ import annotations

import re

import torch

from .. import LATENT_HOP, SAMPLE_RATE
from ..utils.audio import energy_vad, runs

_TAG = re.compile(r"</?[a-z]+>")


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def _clean(s: str) -> str:
    s = _TAG.sub(" ", s.lower())
    s = re.sub(r"[ً-ْٰ]", "", s)  # drop diacritics for scoring
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = _clean(ref).replace(" ", ""), _clean(hyp).replace(" ", "")
    return edit_distance(list(r), list(h)) / max(1, len(r))


def wer(ref: str, hyp: str) -> float:
    r, h = _clean(ref).split(), _clean(hyp).split()
    return edit_distance(r, h) / max(1, len(r))


def islands_from_wav(wav: torch.Tensor, min_sec: float) -> list[tuple[int, int]]:
    vad = energy_vad(wav, SAMPLE_RATE, LATENT_HOP)
    return [(s, e) for s, e in runs(vad) if (e - s) * LATENT_HOP / SAMPLE_RATE >= min_sec]


def island_similarity(spk, src_wav: torch.Tensor, out_wav: torch.Tensor, min_sec: float = 1.5) -> float | None:
    isl = islands_from_wav(src_wav, min_sec)
    if not isl:
        return None
    h = LATENT_HOP
    a = spk.embed_batch([src_wav[s * h : e * h] for s, e in isl])
    b = spk.embed_batch([out_wav[s * h : e * h] for s, e in isl])
    return float((a * b).sum(-1).mean())


def ssp(spk, src_wav: torch.Tensor, out_wav: torch.Tensor, min_sec: float = 0.5) -> float | None:
    isl = islands_from_wav(src_wav, min_sec)
    if len(isl) < 3:
        return None
    h = LATENT_HOP
    a = spk.embed_batch([src_wav[s * h : e * h] for s, e in isl])
    b = spk.embed_batch([out_wav[s * h : e * h] for s, e in isl])
    sa, sb = a @ a.T, b @ b.T
    iu = torch.triu_indices(len(isl), len(isl), 1)
    x, y = sa[iu[0], iu[1]], sb[iu[0], iu[1]]
    if x.std() < 1e-6 or y.std() < 1e-6:
        return None
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def boundary_error(out_wav: torch.Tensor, chunk_frames: list[tuple[int, int]], tol: int = 2) -> float:
    vad = energy_vad(out_wav, SAMPLE_RATE, LATENT_HOP)
    allowed = torch.zeros_like(vad)
    for s, e in chunk_frames:
        allowed[max(0, s - tol) : e + tol] = True
    speech = vad.sum().clamp_min(1)
    return float((vad & ~allowed).sum() / speech)


def slc(src_wav: torch.Tensor, out_wav: torch.Tensor, p: float = 0.2) -> bool:
    a = energy_vad(src_wav, SAMPLE_RATE, LATENT_HOP).sum().item()
    b = energy_vad(out_wav, SAMPLE_RATE, LATENT_HOP).sum().item()
    return a > 0 and (1 - p) * a <= b <= (1 + p) * a


def pause_iou(src_wav: torch.Tensor, out_wav: torch.Tensor) -> float | None:
    a = ~energy_vad(src_wav, SAMPLE_RATE, LATENT_HOP)
    b = ~energy_vad(out_wav, SAMPLE_RATE, LATENT_HOP)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    union = (a | b).sum()
    return None if union == 0 else float((a & b).sum() / union)


class ASR:
    """Optional ASR backend for CER rewards / evaluation (transformers Whisper)."""

    def __init__(self, model: str = "openai/whisper-large-v3", device: str = "cpu"):
        from transformers import pipeline

        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device)

    def __call__(self, wav: torch.Tensor, lang: str) -> str:
        from ..utils.audio import resample

        w = resample(wav.float().cpu(), SAMPLE_RATE, 16000).numpy()
        lang_name = {"fa": "persian", "en": "english", "zh": "chinese"}.get(lang)
        out = self.pipe(w, generate_kwargs={"language": lang_name, "task": "transcribe"} if lang_name else None)
        return out["text"]
