"""Build a compact dataset index (speaker embeddings, lengths, languages) used by
the pretraining dataset for partner / hard-negative selection, and merge
sharded latent statistics.

  python -m dubflow.preprocess.build_index --manifest m.jsonl --cache feats/ --out feats/index.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .extract import read_manifest


def merge_latent_stats(cache_dir: str | Path) -> dict:
    cache_dir = Path(cache_dir)
    single = cache_dir / "latent_stats.pt"
    if single.exists():
        return torch.load(single, weights_only=False)
    shards = sorted(cache_dir.glob("latent_stats.shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no latent_stats in {cache_dir}")
    parts = [torch.load(p, weights_only=False) for p in shards]
    n = sum(p["n"] for p in parts)
    mean = sum(p["mean"] * p["n"] for p in parts) / n
    var = sum((p["std"] ** 2 + (p["mean"] - mean) ** 2) * p["n"] for p in parts) / n
    merged = dict(parts[0])
    merged.update(mean=mean, std=var.sqrt(), n=n)
    torch.save(merged, single)
    return merged


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = read_manifest(args.manifest)
    ids, langs, speakers, frames, embs = [], [], [], [], []
    for row in tqdm(rows, desc="index"):
        p = Path(args.cache) / f"{row['id']}.pt"
        if not p.exists():
            continue
        item = torch.load(p, weights_only=False)
        ids.append(row["id"])
        langs.append(row.get("lang", "other"))
        speakers.append(row.get("speaker", row["id"]))
        frames.append(int(item["n_frames"]))
        embs.append(item["spk_utt"].float())
    torch.save({"ids": ids, "langs": langs, "speakers": speakers, "frames": torch.tensor(frames),
                "spk": torch.stack(embs).half() if embs else torch.zeros(0)}, args.out)
    merge_latent_stats(args.cache)
    print(f"indexed {len(ids)} items -> {args.out}")


if __name__ == "__main__":
    main()
