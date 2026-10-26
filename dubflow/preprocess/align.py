"""Word-level forced alignment for pretraining manifests (adds "words").

Uses torchaudio's MMS_FA (multilingual, 1100+ languages). Non-Latin scripts
(Persian, Chinese characters) must be romanized first; install `uroman`
(`pip install uroman`) and it is used automatically. Chinese should already be
pinyin in the manifest.

  python -m dubflow.preprocess.align --manifest in.jsonl --out out.jsonl --device cuda
If your data already has word timestamps (e.g. from WhisperX), skip this step.
"""

from __future__ import annotations

import argparse
import json
import re

import torch
from tqdm import tqdm

from ..utils.audio import load_audio


def _romanizer():
    try:
        import uroman as ur

        u = ur.Uroman()
        return lambda s: u.romanize_string(s)
    except Exception:
        return None


def _normalize_words(text: str, roman) -> tuple[list[str], list[str]]:
    words = [w for w in re.sub(r"</?[a-z]+>", " ", text).split() if w.strip()]
    normed = []
    for w in words:
        r = roman(w) if roman else w
        r = re.sub(r"[^a-z']", "", r.lower())
        normed.append(r or "a")
    return words, normed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(args.device)
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    roman = _romanizer()
    sr = bundle.sample_rate
    with open(args.manifest, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as fo:
        for line in tqdm(f):
            row = json.loads(line)
            words, normed = _normalize_words(row.get("text", ""), roman if row.get("lang") != "en" else None)
            if words:
                wav = load_audio(row["audio"], sr)[None].to(args.device)
                with torch.inference_mode():
                    emission, _ = model(wav)
                spans = aligner(emission[0], tokenizer(normed))
                ratio = wav.shape[1] / emission.shape[1] / sr
                row["words"] = [{"w": w, "s": round(sp[0].start * ratio, 3), "e": round(sp[-1].end * ratio, 3)}
                                for w, sp in zip(words, spans)]
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
