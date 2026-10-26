"""Feature extraction: audio manifest -> per-item feature cache (.pt).

Manifest (JSONL), one utterance or scene per line:
  {"id": "spk1_0001", "audio": "/data/a.wav", "lang": "en", "text": "...",
   "speaker": "spk1",                                   # optional
   "words": [{"w": "hello", "s": 0.12, "e": 0.48}, ...]} # optional, seconds

Cache file `<out>/<id>.pt` contains (25 Hz frames):
  latent   [T, D] float16  raw VAE posterior mean (normalized at load time)
  f0       [T]   float16  Hz (0 = unvoiced)
  energy   [T]   float16  dB
  vad      [T]   bool
  islands  [K, 2] int     speech-island frame ranges [s, e)
  spk_island [K, Ds] f16  speaker embedding per island
  spk_utt  [Ds]  f16      whole-utterance embedding (speech only)
  spk_short [N5, Ds] f16  optional 5 Hz sliding-window track (--short-window)
  emo      [N5, De] f16   5 Hz emotion / voice-quality track
Also writes `<out>/latent_stats.pt` (mean / std / silence latent).

With `--augment N`, N degraded copies of every item are also written as
`<out>/<id>.aug<k>.pt` (same fields, computed from the degraded audio: noise,
music / effects bleed, reverb, EQ, band limiting, separation artifacts; see
`dubflow.utils.augment`). Training uses them for the conditioning side only
(data.src_aug_p / data.src_aug_versions); they never enter the latent stats.
Real noise / music stems can be given as JSONL lists ({"audio": path}).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .. import LATENT_HOP, SAMPLE_RATE
from ..features.backends import EmotionEmbedder, SpeakerEmbedder
from ..utils.audio import energy_vad, estimate_f0, frame_energy, load_audio, runs
from ..utils.augment import SourceAugmenter
from ..utils.checkpoint import load_vae


def read_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def speech_islands(vad: torch.Tensor, min_frames: int = 5) -> list[tuple[int, int]]:
    return [(s, e) for s, e in runs(vad) if e - s >= min_frames]


@torch.no_grad()
def extract_item(wav: torch.Tensor, vae, spk: SpeakerEmbedder, emo: EmotionEmbedder, device: str, short_window: bool) -> dict:
    hop = LATENT_HOP
    wav = wav[: (wav.numel() // hop) * hop] if wav.numel() >= hop else torch.nn.functional.pad(wav, (0, hop - wav.numel()))
    t = wav.numel() // hop
    _, mu, _ = vae.encode(wav[None].to(device))
    latent = mu[0].T[:t].float().cpu()
    f0, _ = estimate_f0(wav, SAMPLE_RATE, hop)
    energy = frame_energy(wav, hop)
    vad = energy_vad(wav, SAMPLE_RATE, hop)
    f0, energy, vad = f0[:t], energy[:t], vad[:t]
    islands = speech_islands(vad) or [(0, t)]
    segs = [wav[s * hop : e * hop] for s, e in islands]
    spk_island = spk.embed_batch(segs)
    speech = torch.cat(segs) if segs else wav
    spk_utt = spk.embed(speech)
    out = {
        "latent": latent.half(),
        "f0": f0.half(),
        "energy": energy.half(),
        "vad": vad,
        "islands": torch.tensor(islands, dtype=torch.long),
        "spk_island": spk_island.half(),
        "spk_utt": spk_utt.half(),
        "emo": emo.track(wav).half(),
        "n_frames": t,
    }
    if short_window:
        out["spk_short"] = spk.embed_windows(wav, 1.0, 0.2).half()
    return out


class LatentStats:
    def __init__(self, dim: int):
        self.n = 0
        self.s = torch.zeros(dim, dtype=torch.float64)
        self.ss = torch.zeros(dim, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        x = x.double()
        self.n += x.shape[0]
        self.s += x.sum(0)
        self.ss += (x * x).sum(0)

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.s / max(self.n, 1)
        var = self.ss / max(self.n, 1) - mean * mean
        return mean.float(), var.clamp_min(1e-8).sqrt().float()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--vae", required=True, help="DubVAE checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--speaker-backend", default="wavlm_sv")
    ap.add_argument("--speaker-dim", type=int, default=None, help="only for the fallback backend")
    ap.add_argument("--emotion-backend", default="emotion2vec")
    ap.add_argument("--emotion-dim", type=int, default=64)
    ap.add_argument("--short-window", action="store_true", help="also store the 5 Hz sliding speaker track (source/dub data)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--stats-max-items", type=int, default=5000)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--augment", type=int, default=0, help="also write N degraded copies <id>.aug<k>.pt")
    ap.add_argument("--noise-list", default=None, help="JSONL of noise recordings for --augment")
    ap.add_argument("--music-list", default=None, help="JSONL of music / effects stems for --augment")
    ap.add_argument("--aug-seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)[args.shard :: args.num_shards]
    vae = load_vae(args.vae, args.device)
    spk = SpeakerEmbedder(args.speaker_backend, args.device, SAMPLE_RATE, args.speaker_dim)
    emo = EmotionEmbedder(args.emotion_backend, args.device, SAMPLE_RATE, args.emotion_dim)
    stats = LatentStats(vae.latent_dim)
    augmenter = SourceAugmenter(SAMPLE_RATE, args.noise_list, args.music_list) if args.augment else None
    for i, row in enumerate(tqdm(rows, desc=f"extract[{args.shard}]")):
        dst = out / f"{row['id']}.pt"
        aug_dst = [out / f"{row['id']}.aug{k}.pt" for k in range(args.augment)]
        if args.skip_existing and dst.exists() and all(p.exists() for p in aug_dst):
            continue
        wav = load_audio(row["audio"], SAMPLE_RATE)
        if not (args.skip_existing and dst.exists()):
            item = extract_item(wav, vae, spk, emo, args.device, args.short_window)
            torch.save(item, dst)
            if i < args.stats_max_items:
                stats.update(item["latent"].float())
        for k, p in enumerate(aug_dst):
            if args.skip_existing and p.exists():
                continue
            augmenter.rng = random.Random(f"{args.aug_seed}:{row['id']}:{k}")  # reproducible per item
            noisy, info = augmenter(wav)
            aug_item = extract_item(noisy, vae, spk, emo, args.device, args.short_window)
            aug_item["aug"] = info
            torch.save(aug_item, p)
    mean, std = stats.result()
    with torch.no_grad():
        silence = vae.encode(torch.zeros(1, SAMPLE_RATE, device=args.device))[1][0].T.mean(0).float().cpu()
    stats_path = out / ("latent_stats.pt" if args.num_shards == 1 else f"latent_stats.shard{args.shard}.pt")
    torch.save({"mean": mean, "std": std, "silence": silence, "n": stats.n,
                "speaker_backend": args.speaker_backend, "speaker_dim": spk.dim, "emotion_dim": args.emotion_dim}, stats_path)
    print(f"wrote {len(rows)} items and {stats_path}")


if __name__ == "__main__":
    main()
