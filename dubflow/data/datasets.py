"""Training datasets.

PretrainDataset (stage D1): monolingual utterances, optionally composed into
synthetic multi-speaker scenes (incl. hard negatives: similar voices back to
back, sometimes inside one text chunk with no pause). All conditioning comes
from the target itself but is content-free (identity / coarse prosody /
emotion / activity) and the timbre window is offset in time (offset-window).

DubDataset (stage D2): source/target scene pairs on a shared timeline (e.g.
English original -> Persian dub voice-converted to the original actor). All
conditioning comes from the source at the same time (exactly as at inference).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .. import FRAME_RATE, LATENT_HOP, SAMPLE_RATE
from ..text.tokenizer import LANGS, TextTokenizer
from ..utils.audio import load_audio
from .tracks import (
    Chunk,
    activity_track,
    build_frame_tok,
    chunks_from_words,
    degrade_prosody,
    island_track,
    prosody_track,
    resample_track,
    smooth_identity,
    tokenize_chunks,
    window_centers_offset,
)


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _Base(Dataset):
    def __init__(self, stats: dict, data_cfg, n_streams: int = 2):
        self.cfg = data_cfg
        self.mean = stats["mean"].detach().float()
        self.std = stats["std"].detach().float()
        self.silence = ((stats["silence"].detach().float() - self.mean) / self.std)
        self.speaker_dim = int(stats["speaker_dim"])
        self.emotion_dim = int(stats["emotion_dim"])
        self.tok = TextTokenizer()
        self.n_streams = n_streams
        self.max_frames = int(data_cfg.max_frames)

    def norm(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent.float() - self.mean) / self.std

    def _load(self, cache_dir: str, item_id: str) -> dict:
        return torch.load(Path(cache_dir) / f"{item_id}.pt", weights_only=False)

    def _emotion(self, item: dict, n: int) -> torch.Tensor:
        return resample_track(item["emo"].float(), n) if item["emo"].shape[0] else torch.zeros(n, self.emotion_dim)

    def _finish(self, x1, chunks, lang, tracks, src_latent, win_center, loss_mask, wav=None) -> dict:
        ids, spans = tokenize_chunks(chunks, self.tok, lang)
        n = x1.shape[0]
        out = {
            "x1": x1,
            "text": torch.tensor(ids or [self.tok.space_id], dtype=torch.long),
            "frame_tok": build_frame_tok(chunks, spans, n, self.n_streams),
            "lang": LANGS.get(lang, LANGS["other"]),
            "tracks": torch.cat(tracks, -1).float(),
            "src_latent": src_latent,
            "win_center": win_center,
            "loss_mask": loss_mask,
        }
        if wav is not None:
            out["wav"] = wav
        return out


class PretrainDataset(_Base):
    def __init__(self, manifest: str, cache_dir: str, index_path: str, stats: dict, data_cfg,
                 n_streams: int = 2, return_wav: bool = False, seed: int = 0):
        super().__init__(stats, data_cfg, n_streams)
        self.cache_dir = cache_dir
        index = torch.load(index_path, weights_only=False)
        by_id = {r["id"]: r for r in _read_jsonl(manifest)}
        self.rows = [by_id[i] for i in index["ids"]]
        self.langs = index["langs"]
        self.speakers = index["speakers"]
        self.spk = torch.nn.functional.normalize(index["spk"].float(), dim=-1) if len(index["spk"]) else None
        self.by_lang: dict[str, list[int]] = {}
        for i, lang in enumerate(self.langs):
            self.by_lang.setdefault(lang, []).append(i)
        self.return_wav = return_wav
        self.rng = random.Random(seed)
        # Language-balanced sampling weights (e.g. upsample Persian).
        lw = dict(data_cfg.get("lang_weights", {}))
        self.weights = torch.tensor([float(lw.get(lang, 1.0)) for lang in self.langs])

    def __len__(self) -> int:
        return len(self.rows)

    # ----------------------------------------------------------------- helpers
    def _partner(self, i: int) -> int:
        cands = self.by_lang[self.langs[i]]
        pool = [j for j in self.rng.sample(cands, min(len(cands), 64)) if self.speakers[j] != self.speakers[i]]
        if not pool:
            return i
        if self.spk is not None and self.rng.random() < self.cfg.get("hard_negative_p", 0.5):
            sims = self.spk[pool] @ self.spk[i]
            return pool[int(sims.argmax())]
        return self.rng.choice(pool)

    def _crop(self, item: dict, row: dict, max_len: int) -> tuple[int, int]:
        n = int(item["n_frames"])
        if n <= max_len:
            return 0, n
        start = self.rng.randint(0, n - max_len)
        return start, start + max_len

    # ----------------------------------------------------------------- main
    def __getitem__(self, idx: int) -> dict:
        rng = self.rng
        cfg = self.cfg
        lang = self.langs[idx]
        members = [idx]
        if rng.random() < cfg.get("scene_p", 0.5):
            for _ in range(rng.randint(1, cfg.get("max_scene_speakers", 3) - 1)):
                members.append(self._partner(members[-1]))

        latents, chunks, tracks_parts, centers, regions, wavs = [], [], {k: [] for k in ("isl", "short", "pro", "emo", "act")}, [], [], []
        pos = 0
        budget = self.max_frames
        min_off = int(cfg.get("window_min_offset_sec", 3.0) * FRAME_RATE)
        prev_last_chunk_idx = None
        for mi, j in enumerate(members):
            if budget <= 10:
                break
            if mi > 0:
                no_gap = rng.random() < cfg.get("no_gap_p", 0.3)
                gap = 0 if no_gap else rng.randint(1, int(cfg.get("max_gap_sec", 0.8) * FRAME_RATE))
                if budget - gap <= 10:
                    break
                if gap:
                    latents.append(self.silence.expand(gap, -1))
                    for k, d in (("isl", self.speaker_dim), ("short", self.speaker_dim), ("emo", self.emotion_dim)):
                        tracks_parts[k].append(torch.zeros(gap, d))
                    tracks_parts["pro"].append(torch.zeros(gap, 3))
                    tracks_parts["act"].append(torch.zeros(gap, 3))
                    centers.append(torch.full((gap,), -1, dtype=torch.long))
                    if self.return_wav:
                        wavs.append(torch.zeros(gap * LATENT_HOP))
                    pos += gap
                    budget -= gap
            row = self.rows[j]
            item = self._load(self.cache_dir, row["id"])
            s, e = self._crop(item, row, budget)
            n = e - s
            lat = self.norm(item["latent"][s:e])
            latents.append(lat)
            vad = item["vad"][s:e]
            isl = island_track(item["islands"] - s, item["spk_island"], n)
            tracks_parts["isl"].append(isl)
            tracks_parts["short"].append(isl)  # smoothed jointly below (simulated sliding window)
            pro = prosody_track(item["f0"][s:e], item["energy"][s:e], vad)
            islands = [(max(0, a - s), min(n, b - s)) for a, b in item["islands"].tolist() if b > s and a < e]
            tracks_parts["pro"].append(degrade_prosody(pro, rng, islands=islands))
            emo_full = self._emotion(item, int(item["n_frames"]))
            tracks_parts["emo"].append(emo_full[s:e])
            tracks_parts["act"].append(activity_track(vad))
            regions.append((pos, pos + n))
            # Text chunks
            words = [w for w in row.get("words", []) if w["s"] * FRAME_RATE >= s and w["e"] * FRAME_RATE <= e]
            words = [{"w": w["w"], "s": w["s"] - s / FRAME_RATE, "e": w["e"] - s / FRAME_RATE} for w in words]
            new_chunks = chunks_from_words(words, FRAME_RATE, pos, rng)
            if not new_chunks:
                speech = vad.nonzero()
                a = int(speech[0]) if len(speech) else 0
                b = int(speech[-1]) + 1 if len(speech) else n
                new_chunks = [Chunk(pos + a, pos + b, row.get("text", ""))]
            # Hard case: merge across a no-gap speaker change into ONE chunk.
            if (mi > 0 and prev_last_chunk_idx is not None and latents and gap == 0
                    and rng.random() < cfg.get("cross_speaker_chunk_p", 0.5)):
                first = new_chunks.pop(0)
                last = chunks[prev_last_chunk_idx]
                chunks[prev_last_chunk_idx] = Chunk(last.start, first.end, f"{last.text} {first.text}")
            chunks.extend(new_chunks)
            prev_last_chunk_idx = len(chunks) - 1
            if self.return_wav:
                wav = load_audio(row["audio"], SAMPLE_RATE)[s * LATENT_HOP : e * LATENT_HOP]
                wavs.append(torch.nn.functional.pad(wav, (0, n * LATENT_HOP - wav.numel())))
            pos += n
            budget -= n

        x1 = torch.cat(latents, 0)
        total = x1.shape[0]
        centers_t = window_centers_offset(regions, total, min_off, rng)
        short = smooth_identity(torch.cat(tracks_parts["short"], 0), int(FRAME_RATE))
        tracks = [torch.cat(tracks_parts["isl"], 0), short, torch.cat(tracks_parts["pro"], 0),
                  torch.cat(tracks_parts["emo"], 0), torch.cat(tracks_parts["act"], 0)]
        wav = torch.cat(wavs) if self.return_wav else None
        return self._finish(x1, chunks, lang, tracks, x1.clone(), centers_t,
                            torch.ones(total, dtype=torch.bool), wav)


class DubDataset(_Base):
    """Rows: {"id", "src_id", "tgt_id", "src_lang", "tgt_lang",
              "chunks": [{"t0", "t1", "text", "stream": 0, "ok": true}]}  (seconds)"""

    def __init__(self, manifest: str, src_cache: str, tgt_cache: str, stats: dict, data_cfg,
                 n_streams: int = 2, return_wav: bool = False, tgt_audio_key: str = "tgt_audio"):
        super().__init__(stats, data_cfg, n_streams)
        self.rows = _read_jsonl(manifest)
        self.src_cache, self.tgt_cache = src_cache, tgt_cache
        self.return_wav = return_wav
        self.tgt_audio_key = tgt_audio_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        src = self._load(self.src_cache, row["src_id"])
        tgt = self._load(self.tgt_cache, row["tgt_id"])
        n = min(int(src["n_frames"]), int(tgt["n_frames"]), self.max_frames)
        x1 = self.norm(tgt["latent"][:n])
        src_lat = self.norm(src["latent"][:n])
        vad = src["vad"][:n]
        isl = island_track(src["islands"], src["spk_island"], n)
        if "spk_short" in src:
            short = resample_track(src["spk_short"].float(), int(src["n_frames"]))[:n]
        else:
            short = smooth_identity(isl, int(FRAME_RATE))
        pro = prosody_track(src["f0"][:n], src["energy"][:n], vad)
        emo = self._emotion(src, int(src["n_frames"]))[:n]
        act = activity_track(vad)
        chunks, loss_mask = [], torch.ones(n, dtype=torch.bool)
        for c in row["chunks"]:
            s, e = int(round(c["t0"] * FRAME_RATE)), int(round(c["t1"] * FRAME_RATE))
            if s >= n:
                continue
            ch = Chunk(s, min(e, n), c["text"], int(c.get("stream", 0)), bool(c.get("ok", True)))
            chunks.append(ch)
            if not ch.ok:
                loss_mask[ch.start : ch.end] = False
        wav = None
        if self.return_wav and self.tgt_audio_key in row:
            w = load_audio(row[self.tgt_audio_key], SAMPLE_RATE)[: n * LATENT_HOP]
            wav = torch.nn.functional.pad(w, (0, n * LATENT_HOP - w.numel()))
        centers = torch.arange(n)
        return self._finish(x1, chunks, row.get("tgt_lang", "fa"), [isl, short, pro, emo, act],
                            src_lat, centers, loss_mask, wav)


def collate(items: list[dict]) -> dict:
    b = len(items)
    t = max(it["x1"].shape[0] for it in items)
    ts = max(it["src_latent"].shape[0] for it in items)
    nmax = max(it["text"].shape[0] for it in items)
    d = items[0]["x1"].shape[1]
    c = items[0]["tracks"].shape[1]
    s = items[0]["frame_tok"].shape[0]
    out = {
        "x1": torch.zeros(b, t, d),
        "frame_mask": torch.zeros(b, t, dtype=torch.bool),
        "loss_mask": torch.zeros(b, t, dtype=torch.bool),
        "text": torch.zeros(b, nmax, dtype=torch.long),
        "text_mask": torch.zeros(b, nmax, dtype=torch.bool),
        "frame_tok": torch.full((b, s, t), -1, dtype=torch.long),
        "lang": torch.tensor([it["lang"] for it in items], dtype=torch.long),
        "tracks": torch.zeros(b, t, c),
        "src_latent": torch.zeros(b, ts, d),
        "src_mask": torch.zeros(b, ts, dtype=torch.bool),
        "win_center": torch.full((b, t), -1, dtype=torch.long),
    }
    has_wav = all("wav" in it for it in items)
    if has_wav:
        out["wav"] = torch.zeros(b, t * LATENT_HOP)
    for i, it in enumerate(items):
        n, ns, nt = it["x1"].shape[0], it["src_latent"].shape[0], it["text"].shape[0]
        out["x1"][i, :n] = it["x1"]
        out["frame_mask"][i, :n] = True
        out["loss_mask"][i, :n] = it["loss_mask"]
        out["text"][i, :nt] = it["text"]
        out["text_mask"][i, :nt] = True
        out["frame_tok"][i, :, :n] = it["frame_tok"]
        out["tracks"][i, :n] = it["tracks"]
        out["src_latent"][i, :ns] = it["src_latent"]
        out["src_mask"][i, :ns] = True
        out["win_center"][i, :n] = it["win_center"]
        if has_wav:
            out["wav"][i, : it["wav"].numel()] = it["wav"]
    return out
