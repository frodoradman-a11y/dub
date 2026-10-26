"""Dub a full program: original vocals + time-anchored target script -> dubbed vocals
(optionally mixed with the music & effects stem).

Script JSON (produced by the external translator):
  {"lang": "fa",
   "chunks": [{"t0": 12.40, "t1": 14.10, "text": "...", "stream": 0}, ...]}   # seconds, absolute

  dubflow-dub --model runs/dubflow_m_d3/last.pt --vae runs/vae/vae_0800000.pt \
      --src film_vocals_en.wav --script film_fa.json --bg film_music_effects.wav --out film_fa.wav
"""

from __future__ import annotations

import argparse
import json

import torch

from .. import LATENT_HOP, SAMPLE_RATE
from ..features.backends import EmotionEmbedder, SpeakerEmbedder
from ..utils.audio import load_audio, save_audio
from ..utils.checkpoint import load_checkpoint, load_dubflow, load_vae
from .scene import build_scene_batch, match_loudness, split_scenes


@torch.no_grad()
def dub(model, vae, stats: dict, src: torch.Tensor, script: dict, spk, emo, device, steps: int, guidance: dict,
        max_scene_sec: float, seed: int = 0, step_cond: bool = False, loudness: bool = True) -> torch.Tensor:
    lang = script.get("lang", "fa")
    chunks = sorted(script["chunks"], key=lambda c: c["t0"])
    scenes = split_scenes(src, chunks, max_scene_sec)
    out = torch.zeros_like(src)
    hop = LATENT_HOP
    for si, (fs, fe) in enumerate(scenes):
        t0, t1 = fs * hop / SAMPLE_RATE, fe * hop / SAMPLE_RATE
        scene_chunks = [dict(c, t0=c["t0"] - t0, t1=min(c["t1"], t1) - t0) for c in chunks if t0 <= c["t0"] < t1]
        seg = src[fs * hop : fe * hop]
        if not scene_chunks or seg.numel() < hop:
            continue
        batch, _ = build_scene_batch(seg, scene_chunks, lang, vae, spk, emo, stats, model.n_streams, device)
        x = model.sample(batch, steps=steps, guidance=guidance, sway=-1.0 if not step_cond else 0.0,
                         seed=seed + si, step_cond=step_cond)
        z = x * stats["std"].to(device) + stats["mean"].to(device)
        wav = vae.decode(z.transpose(1, 2))[0].float().cpu()[: seg.numel()]
        if loudness:
            spans = [(int(c["t0"] * SAMPLE_RATE), int(c["t1"] * SAMPLE_RATE)) for c in scene_chunks]
            wav = match_loudness(wav, seg, spans)
        fade = min(240, wav.numel() // 4)
        if fade:
            ramp = torch.linspace(0, 1, fade)
            wav[:fade] *= ramp
            wav[-fade:] *= ramp.flip(0)
        out[fs * hop : fs * hop + wav.numel()] = wav
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--src", required=True, help="original vocals (separated), any sample rate")
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg", default=None, help="music & effects stem to mix under the dub")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance-all", type=float, default=None)
    ap.add_argument("--guidance-timbre", type=float, default=None)
    ap.add_argument("--max-scene-sec", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speaker-backend", default="wavlm_sv")
    ap.add_argument("--emotion-backend", default="emotion2vec")
    ap.add_argument("--no-loudness-match", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, cfg = load_dubflow(args.model, device)
    ck = load_checkpoint(args.model)
    stats = ck["latent_stats"]
    step_cond = bool(cfg.model.get("step_cond", False))
    steps = args.steps or ck.get("student_steps") or cfg.sample.steps
    guidance = dict(cfg.sample.guidance) if not step_cond else {}
    if args.guidance_all is not None:
        guidance["all"] = args.guidance_all
    if args.guidance_timbre is not None:
        guidance["timbre"] = args.guidance_timbre
    vae = load_vae(args.vae, device)
    spk = SpeakerEmbedder(args.speaker_backend, str(device), SAMPLE_RATE, int(stats["speaker_dim"]))
    if spk.dim != int(stats["speaker_dim"]):
        raise ValueError(f"speaker backend dim {spk.dim} != model {stats['speaker_dim']}")
    emo = EmotionEmbedder(args.emotion_backend, str(device), SAMPLE_RATE, int(stats["emotion_dim"]))
    with open(args.script, encoding="utf-8") as f:
        script = json.load(f)
    src = load_audio(args.src, SAMPLE_RATE)
    max_sec = args.max_scene_sec or cfg.data.max_frames / 25.0
    out = dub(model, vae, stats, src, script, spk, emo, device, steps, guidance, max_sec, args.seed, step_cond,
              not args.no_loudness_match)
    if args.bg:
        bg = load_audio(args.bg, SAMPLE_RATE)
        n = max(out.numel(), bg.numel())
        out = torch.nn.functional.pad(out, (0, n - out.numel())) + torch.nn.functional.pad(bg, (0, n - bg.numel()))
    peak = out.abs().max()
    if peak > 0.99:
        out = out * (0.99 / peak)
    save_audio(args.out, out, SAMPLE_RATE)
    print(f"wrote {args.out} ({out.numel() / SAMPLE_RATE:.1f}s)")


if __name__ == "__main__":
    main()
