"""Training datasets.

PretrainDataset (stage D1): monolingual utterances, optionally composed into
synthetic multi-speaker scenes (incl. hard negatives: similar voices back to
back, sometimes inside one text chunk with no pause, and overlapping speech
mixed at waveform level). All conditioning comes from the target itself but is
content-free (identity / coarse prosody / emotion / activity) and the timbre
window is offset in time (offset-window).

DubDataset (stage D2): source/target scene pairs on a shared timeline (e.g.
English original -> Persian dub voice-converted to the original actor). All
conditioning comes from the source at the same time (exactly as at inference).

Shared by both:
- source augmentation: with probability `src_aug_p` the conditioning (source
  latent + tracks) is read from a degraded copy of the source
  (`<id>.aug<k>.pt`, written by `extract --augment`); the target stays clean.
- scene continuity: with probability `ctx_p` a prefix of the target is given as
  known context (`ctx_mask`, excluded from the loss), as when inference
  continues from the end of the previous scene.
"""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import torch
from torch.utils.data import Dataset, get_worker_info

from .. import FRAME_RATE, LATENT_HOP, SAMPLE_RATE
from ..text.tokenizer import LANGS, TextTokenizer
from ..utils.audio import load_audio
from .tracks import (
    TRACK_ORDER,
    Chunk,
    activity_track,
    assign_streams,
    build_frame_tok,
    chunks_from_words,
    degrade_prosody,
    island_track,
    overlap_tracks,
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
    def __init__(self, stats: dict, data_cfg, n_streams: int = 2, seed: int = 0):
        self.cfg = data_cfg
        self.mean = stats["mean"].detach().float()
        self.std = stats["std"].detach().float()
        self.silence = ((stats["silence"].detach().float() - self.mean) / self.std)
        self.speaker_dim = int(stats["speaker_dim"])
        self.emotion_dim = int(stats["emotion_dim"])
        self.tok = TextTokenizer()
        self.n_streams = n_streams
        self.max_frames = int(data_cfg.max_frames)
        self.seed = seed
        self.rng = random.Random(seed)
        self._worker: int | None = None
        self.aug_p = float(data_cfg.get("src_aug_p", 0.0))
        self.aug_versions = int(data_cfg.get("src_aug_versions", 0))
        self._aug_missing_warned = False

    def _reseed(self) -> None:
        """DataLoader workers get a copy of the dataset (and of self.rng): give
        each worker its own random stream."""
        info = get_worker_info()
        wid = info.id if info is not None else -1
        if wid != self._worker:
            self._worker = wid
            if info is not None:
                self.rng.seed(self.seed * 1009 + wid + 1)

    def norm(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent.float() - self.mean) / self.std

    def _load(self, cache_dir: str, item_id: str) -> dict:
        return torch.load(Path(cache_dir) / f"{item_id}.pt", weights_only=False)

    def _want_aug(self) -> bool:
        return self.aug_versions > 0 and self.aug_p > 0 and self.rng.random() < self.aug_p

    def _cond_item(self, cache_dir: str, item_id: str, clean: dict, use_aug: bool) -> dict:
        """Item the conditioning is computed from: a degraded copy of the source
        (if requested and cached) or the clean one."""
        if not use_aug:
            return clean
        p = Path(cache_dir) / f"{item_id}.aug{self.rng.randrange(self.aug_versions)}.pt"
        if p.exists():
            return torch.load(p, weights_only=False)
        if not self._aug_missing_warned:
            warnings.warn(f"src_aug_p > 0 but {p} is missing (run extract --augment); using clean sources")
            self._aug_missing_warned = True
        return clean

    def _emotion(self, item: dict, n: int) -> torch.Tensor:
        return resample_track(item["emo"].float(), n) if item["emo"].shape[0] else torch.zeros(n, self.emotion_dim)

    def _context(self, n: int, chunks: list[Chunk], loss_mask: torch.Tensor):
        """Known-prefix (scene continuity) training: the first k frames are given
        as context, not predicted. Text of chunks that lie entirely inside the
        prefix is dropped half of the time (at inference the previous scene's
        lines may or may not be repeated in the context window)."""
        ctx = torch.zeros(n, dtype=torch.bool)
        p = float(self.cfg.get("ctx_p", 0.0))
        if p <= 0 or self.rng.random() >= p:
            return ctx, chunks, loss_mask
        lo = max(1, int(self.cfg.get("ctx_min_sec", 0.5) * FRAME_RATE))
        hi = min(int(self.cfg.get("ctx_max_sec", 4.0) * FRAME_RATE), n // 2)
        if hi < lo:
            return ctx, chunks, loss_mask
        k = self.rng.randint(lo, hi)
        ctx[:k] = True
        if self.rng.random() < 0.5:
            chunks = [c for c in chunks if c.end > k]
        return ctx, chunks, loss_mask & ~ctx

    def _finish(self, x1, chunks, lang, tracks: dict, src_latent, win_center, loss_mask, wav=None,
                reencode: bool = False) -> dict:
        n = x1.shape[0]
        ctx, chunks, loss_mask = self._context(n, chunks, loss_mask)
        ids, spans = tokenize_chunks(chunks, self.tok, lang)
        out = {
            "x1": x1,
            "text": torch.tensor(ids or [self.tok.space_id], dtype=torch.long),
            "frame_tok": build_frame_tok(chunks, spans, n, self.n_streams),
            "lang": LANGS.get(lang, LANGS["other"]),
            "tracks": torch.cat([tracks[k] for k in TRACK_ORDER], -1).float(),
            "src_latent": src_latent,
            "win_center": win_center,
            "loss_mask": loss_mask,
            "ctx_mask": ctx,
            "reencode": reencode,
        }
        if wav is not None:
            out["wav"] = wav
        return out


def _last_speech(vad: torch.Tensor) -> int:
    idx = vad.nonzero()
    return int(idx[-1]) + 1 if len(idx) else vad.numel()


class PretrainDataset(_Base):
    def __init__(self, manifest: str, cache_dir: str, index_path: str, stats: dict, data_cfg,
                 n_streams: int = 2, return_wav: bool = False, seed: int = 0):
        super().__init__(stats, data_cfg, n_streams, seed)
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

    def _crop(self, item: dict, max_len: int) -> tuple[int, int]:
        n = int(item["n_frames"])
        if n <= max_len:
            return 0, n
        start = self.rng.randint(0, n - max_len)
        return start, start + max_len

    # ----------------------------------------------------------------- main
    def __getitem__(self, idx: int) -> dict:
        self._reseed()
        rng = self.rng
        cfg = self.cfg
        lang = self.langs[idx]
        members = [idx]
        if rng.random() < cfg.get("scene_p", 0.5):
            for _ in range(rng.randint(1, cfg.get("max_scene_speakers", 3) - 1)):
                members.append(self._partner(members[-1]))
        # Overlapping speech: at one speaker change the next speaker starts before
        # the previous one has finished. The mix only exists as a waveform, so
        # the trainer re-encodes such scenes with the VAE ("reencode").
        ov_at = rng.randint(1, len(members) - 1) if len(members) > 1 and rng.random() < cfg.get("overlap_p", 0.0) else -1
        need_wav = self.return_wav or ov_at > 0
        use_aug = ov_at < 0 and self._want_aug()
        min_ov = max(1, int(cfg.get("overlap_min_sec", 0.2) * FRAME_RATE))
        max_ov = max(min_ov, int(cfg.get("overlap_max_sec", 1.5) * FRAME_RATE))

        pieces: list[dict] = []
        chunks: list[Chunk] = []
        pos = 0  # end of the timeline so far
        prev_last_chunk_idx = None
        overlapped = False
        for mi, j in enumerate(members):
            if self.max_frames - pos <= 10:
                break
            gap, ov = 0, 0
            start = pos
            if mi > 0:
                prev = pieces[-1]
                if mi == ov_at:
                    speech_end = prev["start"] + _last_speech(prev["vad"])
                    ov = min(rng.randint(min_ov, max_ov), _last_speech(prev["vad"]) // 2)
                    start = min(pos, speech_end - ov) if ov > 0 else pos
                elif rng.random() >= cfg.get("no_gap_p", 0.3):
                    gap = rng.randint(1, int(cfg.get("max_gap_sec", 0.8) * FRAME_RATE))
                    start = pos + gap
                if self.max_frames - start <= 10:
                    break
            row = self.rows[j]
            item = self._load(self.cache_dir, row["id"])
            s, e = self._crop(item, self.max_frames - start)
            n = e - s
            cond = self._cond_item(self.cache_dir, row["id"], item, use_aug)
            vad = cond["vad"][s:e]
            islands = [(max(0, a - s), min(n, b - s)) for a, b in cond["islands"].tolist() if b > s and a < e]
            pro = prosody_track(cond["f0"][s:e], cond["energy"][s:e], vad)
            piece = {
                "start": start, "n": n,
                "lat": self.norm(item["latent"][s:e]),
                "src": self.norm(cond["latent"][s:e]),
                "isl": island_track(cond["islands"] - s, cond["spk_island"], n),
                "pro": degrade_prosody(pro, rng, islands=islands),
                "emo": self._emotion(cond, int(item["n_frames"]))[s:e],
                "vad": vad,
                "energy": cond["energy"][s:e].float(),
            }
            if need_wav:
                w = load_audio(row["audio"], SAMPLE_RATE)[s * LATENT_HOP : e * LATENT_HOP]
                piece["wav"] = torch.nn.functional.pad(w, (0, n * LATENT_HOP - w.numel()))
            # Text chunks
            words = [w for w in row.get("words", []) if w["s"] * FRAME_RATE >= s and w["e"] * FRAME_RATE <= e]
            words = [{"w": w["w"], "s": w["s"] - s / FRAME_RATE, "e": w["e"] - s / FRAME_RATE} for w in words]
            new_chunks = chunks_from_words(words, FRAME_RATE, start, rng)
            if not new_chunks:
                clean_vad = item["vad"][s:e]
                speech = clean_vad.nonzero()
                a = int(speech[0]) if len(speech) else 0
                b = int(speech[-1]) + 1 if len(speech) else n
                new_chunks = [Chunk(start + a, start + b, row.get("text", ""))]
            # Hard case: merge across a no-gap speaker change into ONE chunk.
            if (mi > 0 and prev_last_chunk_idx is not None and gap == 0 and start == pos
                    and rng.random() < cfg.get("cross_speaker_chunk_p", 0.5)):
                first = new_chunks.pop(0)
                last = chunks[prev_last_chunk_idx]
                chunks[prev_last_chunk_idx] = Chunk(last.start, first.end, f"{last.text} {first.text}")
            chunks.extend(new_chunks)
            prev_last_chunk_idx = len(chunks) - 1
            overlapped |= start < pos
            pieces.append(piece)
            pos = max(pos, start + n)

        total = pos
        x1 = self.silence.expand(total, -1).clone()
        src = x1.clone()
        isl = torch.zeros(total, self.speaker_dim)
        isl_cnt = torch.zeros(total)
        emo = torch.zeros(total, self.emotion_dim)
        emo_cnt = torch.zeros(total, 1)
        pro = torch.zeros(total, 3)
        loudest = torch.full((total,), -1e9)
        vad = torch.zeros(total, dtype=torch.bool)
        wav = torch.zeros(total * LATENT_HOP) if need_wav else None
        regions = []
        for p in pieces:
            a, b = p["start"], p["start"] + p["n"]
            x1[a:b] = p["lat"]  # in overlaps a placeholder: re-encoded from the mixed wav
            src[a:b] = p["src"]
            isl[a:b] += p["isl"]
            isl_cnt[a:b] += (p["isl"].norm(dim=-1) > 0).float()
            emo[a:b] += p["emo"]
            emo_cnt[a:b] += 1
            louder = p["energy"] > loudest[a:b]  # prosody of the dominant voice
            pro[a:b] = torch.where(louder[:, None], p["pro"], pro[a:b])
            loudest[a:b] = torch.maximum(loudest[a:b], p["energy"])
            vad[a:b] |= p["vad"]
            if wav is not None:
                wav[a * LATENT_HOP : b * LATENT_HOP] += p["wav"]
            regions.append((a, b))
        mixed = isl_cnt > 1  # a "real" embedding of a two-voice frame is a blend
        isl[mixed] = torch.nn.functional.normalize(isl[mixed], dim=-1)
        emo = emo / emo_cnt.clamp_min(1)
        short = smooth_identity(isl, int(FRAME_RATE))  # simulated sliding-window track
        chunks = assign_streams(chunks, self.n_streams)
        spk_ov, ov_flag = overlap_tracks(chunks, short, total, self.n_streams)
        tracks = {"spk_island": isl, "spk_short": short, "spk_overlap": spk_ov, "prosody": pro,
                  "emotion": emo, "activity": activity_track(vad, overlap=ov_flag)}
        min_off = int(cfg.get("window_min_offset_sec", 3.0) * FRAME_RATE)
        centers = window_centers_offset(regions, total, min_off, rng)
        return self._finish(x1, chunks, lang, tracks, src, centers, torch.ones(total, dtype=torch.bool), wav,
                            reencode=overlapped)


class DubDataset(_Base):
    """Rows: {"id", "src_id", "tgt_id", "src_lang", "tgt_lang",
              "chunks": [{"t0", "t1", "text", "stream": 0, "ok": true}]}  (seconds)"""

    def __init__(self, manifest: str, src_cache: str, tgt_cache: str, stats: dict, data_cfg,
                 n_streams: int = 2, return_wav: bool = False, tgt_audio_key: str = "tgt_audio", seed: int = 0):
        super().__init__(stats, data_cfg, n_streams, seed)
        self.rows = _read_jsonl(manifest)
        self.src_cache, self.tgt_cache = src_cache, tgt_cache
        self.return_wav = return_wav
        self.tgt_audio_key = tgt_audio_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        self._reseed()
        row = self.rows[idx]
        src_clean = self._load(self.src_cache, row["src_id"])
        tgt = self._load(self.tgt_cache, row["tgt_id"])
        src = self._cond_item(self.src_cache, row["src_id"], src_clean, self._want_aug())
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
        chunks, loss_mask = [], torch.ones(n, dtype=torch.bool)
        for c in row["chunks"]:
            s, e = int(round(c["t0"] * FRAME_RATE)), int(round(c["t1"] * FRAME_RATE))
            if s >= n:
                continue
            ch = Chunk(s, max(min(e, n), s + 1), c["text"], int(c.get("stream", 0)), bool(c.get("ok", True)))
            chunks.append(ch)
            if not ch.ok:
                loss_mask[ch.start : ch.end] = False
        chunks = assign_streams(chunks, self.n_streams)
        spk_ov, ov_flag = overlap_tracks(chunks, short, n, self.n_streams)
        tracks = {"spk_island": isl, "spk_short": short, "spk_overlap": spk_ov, "prosody": pro,
                  "emotion": emo, "activity": activity_track(vad, overlap=ov_flag)}
        wav = None
        if self.return_wav and self.tgt_audio_key in row:
            w = load_audio(row[self.tgt_audio_key], SAMPLE_RATE)[: n * LATENT_HOP]
            wav = torch.nn.functional.pad(w, (0, n * LATENT_HOP - w.numel()))
        return self._finish(x1, chunks, row.get("tgt_lang", "fa"), tracks, src_lat, torch.arange(n), loss_mask, wav)


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
        "ctx_mask": torch.zeros(b, t, dtype=torch.bool),
        "text": torch.zeros(b, nmax, dtype=torch.long),
        "text_mask": torch.zeros(b, nmax, dtype=torch.bool),
        "frame_tok": torch.full((b, s, t), -1, dtype=torch.long),
        "lang": torch.tensor([it["lang"] for it in items], dtype=torch.long),
        "tracks": torch.zeros(b, t, c),
        "src_latent": torch.zeros(b, ts, d),
        "src_mask": torch.zeros(b, ts, dtype=torch.bool),
        "win_center": torch.full((b, t), -1, dtype=torch.long),
        "reencode": torch.tensor([bool(it.get("reencode", False)) for it in items], dtype=torch.bool),
    }
    if any("wav" in it for it in items):
        out["wav"] = torch.zeros(b, t * LATENT_HOP)
        out["wav_ok"] = torch.tensor(["wav" in it for it in items], dtype=torch.bool)
    for i, it in enumerate(items):
        n, ns, nt = it["x1"].shape[0], it["src_latent"].shape[0], it["text"].shape[0]
        out["x1"][i, :n] = it["x1"]
        out["frame_mask"][i, :n] = True
        out["loss_mask"][i, :n] = it["loss_mask"]
        if "ctx_mask" in it:
            out["ctx_mask"][i, :n] = it["ctx_mask"]
        out["text"][i, :nt] = it["text"]
        out["text_mask"][i, :nt] = True
        out["frame_tok"][i, :, :n] = it["frame_tok"]
        out["tracks"][i, :n] = it["tracks"]
        out["src_latent"][i, :ns] = it["src_latent"]
        out["src_mask"][i, :ns] = True
        out["win_center"][i, :n] = it["win_center"]
        if "wav" in it:
            out["wav"][i, : it["wav"].numel()] = it["wav"]
    return out
