"""Cut aligned source/target programs into training scene pairs (stage D2 data).

Inputs per film (all on the SAME timeline):
  --src-audio   original-language vocals (e.g. English, separated from M&E)
  --tgt-audio   target-language vocals, voice-converted to the original actors
                (e.g. Persian dub -> original voices)
  --tgt-script  time-synced target text: JSON {"chunks": [{"t0","t1","text","ok"?}]}
                or an .srt file (each cue = one chunk)

Outputs:
  <out>/wav/<film>_<k>_src.wav / _tgt.wav       scene audio
  <out>/src_manifest.jsonl, tgt_manifest.jsonl   for `dubflow-extract`
  <out>/pairs.jsonl                              for DubDataset (data.dub_manifest)

  dubflow-build-dub-pairs --film f001 --src-audio en.wav --tgt-audio fa_vc.wav \
      --tgt-script fa.json --src-lang en --tgt-lang fa --out data/dub
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .. import LATENT_HOP, SAMPLE_RATE
from ..utils.audio import energy_vad, load_audio, save_audio

_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    chunks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [x for x in block.splitlines() if x.strip()]
        for i, line in enumerate(lines):
            m = _SRT_TIME.search(line)
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
                t0 = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                t1 = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
                body = " ".join(lines[i + 1 :]).strip()
                if body:
                    chunks.append({"t0": t0, "t1": t1, "text": body})
                break
    return chunks


def load_script(path: str) -> list[dict]:
    if path.lower().endswith(".srt"):
        return parse_srt(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"] if isinstance(data, dict) else data


def cut_points(src: torch.Tensor, tgt: torch.Tensor, chunks: list[dict], max_sec: float, min_sec: float) -> list[tuple[int, int]]:
    """Scene boundaries (frames) where BOTH tracks are silent and no chunk is active."""
    fr = SAMPLE_RATE / LATENT_HOP
    a = energy_vad(src, SAMPLE_RATE, LATENT_HOP)
    b = energy_vad(tgt, SAMPLE_RATE, LATENT_HOP)
    n = min(len(a), len(b))
    busy = a[:n] | b[:n]
    for c in chunks:
        busy[int(c["t0"] * fr) : int(c["t1"] * fr) + 1] = True
    max_f, min_f = int(max_sec * fr), int(min_sec * fr)
    out, start = [], 0
    while start < n - min_f:
        hi = min(n, start + max_f)
        if hi == n:
            out.append((start, n))
            break
        free = (~busy[start + min_f : hi]).nonzero()
        if not len(free):  # no clean cut: skip ahead to the next free frame
            nxt = (~busy[hi:]).nonzero()
            if not len(nxt):
                break
            start = hi + int(nxt[0])
            continue
        cut = start + min_f + int(free[-1])
        out.append((start, cut))
        start = cut
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--film", required=True)
    ap.add_argument("--src-audio", required=True)
    ap.add_argument("--tgt-audio", required=True)
    ap.add_argument("--tgt-script", required=True)
    ap.add_argument("--src-lang", default="en")
    ap.add_argument("--tgt-lang", default="fa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sec", type=float, default=30.0)
    ap.add_argument("--min-sec", type=float, default=4.0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    src = load_audio(args.src_audio, SAMPLE_RATE)
    tgt = load_audio(args.tgt_audio, SAMPLE_RATE)
    chunks = sorted(load_script(args.tgt_script), key=lambda c: c["t0"])
    scenes = cut_points(src, tgt, chunks, args.max_sec, args.min_sec)
    hop = LATENT_HOP
    n_written = 0
    with open(out / "pairs.jsonl", "a", encoding="utf-8") as fp, \
         open(out / "src_manifest.jsonl", "a", encoding="utf-8") as fs, \
         open(out / "tgt_manifest.jsonl", "a", encoding="utf-8") as ft:
        for k, (a, b) in enumerate(scenes):
            t0, t1 = a * hop / SAMPLE_RATE, b * hop / SAMPLE_RATE
            sc = [dict(c, t0=round(c["t0"] - t0, 3), t1=round(min(c["t1"], t1) - t0, 3))
                  for c in chunks if t0 <= c["t0"] < t1]
            if not sc:
                continue
            sid = f"{args.film}_{k:05d}"
            sp, tp = out / "wav" / f"{sid}_src.wav", out / "wav" / f"{sid}_tgt.wav"
            save_audio(str(sp), src[a * hop : b * hop], SAMPLE_RATE)
            save_audio(str(tp), tgt[a * hop : b * hop], SAMPLE_RATE)
            fs.write(json.dumps({"id": f"{sid}_src", "audio": str(sp), "lang": args.src_lang}, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": f"{sid}_tgt", "audio": str(tp), "lang": args.tgt_lang}, ensure_ascii=False) + "\n")
            fp.write(json.dumps({"id": sid, "src_id": f"{sid}_src", "tgt_id": f"{sid}_tgt", "src_lang": args.src_lang,
                                 "tgt_lang": args.tgt_lang, "tgt_audio": str(tp), "chunks": sc}, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"{args.film}: {n_written} scenes")


if __name__ == "__main__":
    main()
