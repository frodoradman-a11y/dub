"""Dub a full program: original vocals + time-anchored target script -> dubbed vocals
(optionally mixed with the music & effects stem).

Script JSON (produced by the external translator):
  {"lang": "fa",
   "chunks": [{"t0": 12.40, "t1": 14.10, "text": "...", "stream": 0}, ...]}   # seconds, absolute
"stream" is optional: lines that overlap in time (two people talking at once)
are put on different streams automatically.

  dubflow-dub --model runs/dubflow_m_d3/last.pt --vae runs/vae/vae_0800000.pt \
      --src film_vocals_en.wav --script film_fa.json --bg film_music_effects.wav --out film_fa.wav
"""

from __future__ import annotations

import argparse
import json

import torch

from .. import FRAME_RATE, LATENT_HOP, SAMPLE_RATE
from ..features.backends import EmotionEmbedder, SpeakerEmbedder
from ..utils.audio import load_audio, save_audio
from ..utils.checkpoint import load_checkpoint, load_dubflow, load_vae
from .acoustics import apply_room, decay_stats, fit_room, ltas_bands, match_eq
from .scene import build_scene_batch, match_loudness, split_scenes


TAIL_SEC = 1.0  # room-reverb tail carried past the end of a scene (overlap-added)


@torch.no_grad()
def dub(model, vae, stats: dict, src: torch.Tensor, script: dict, spk, emo, device, steps: int, guidance: dict,
        max_scene_sec: float, seed: int = 0, step_cond: bool = False, loudness: bool = True, acoustic: bool = True,
        context_sec: float = 0.0) -> torch.Tensor:
    """Scene by scene: generate -> per-line loudness match -> EQ match -> room
    match (reverb tail overlap-added into the next scene). With context_sec > 0
    (and a model trained with the context channel) each scene's window starts
    `context_sec` before the scene and those frames are fixed to the previous
    scene's generated latent, so voice / level / pace continue across the cut."""
    lang = script.get("lang", "fa")
    chunks = sorted(script["chunks"], key=lambda c: c["t0"])
    scenes = split_scenes(src, chunks, max_scene_sec)
    hop = LATENT_HOP
    out = torch.zeros(src.numel() + int(TAIL_SEC * SAMPLE_RATE))
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    # Whole-program fallbacks when a scene has too little speech to measure.
    film_bands = ltas_bands(src, SAMPLE_RATE) if acoustic else None
    film_room = decay_stats(src, SAMPLE_RATE) if acoustic else None
    k_max = int(context_sec * FRAME_RATE) if getattr(model, "ctx_enabled", False) else 0
    prev = None  # (end frame of the previous window, its normalized latent [T, D])
    for si, (fs, fe) in enumerate(scenes):
        k = min(k_max, prev[1].shape[0], fs) if (k_max and prev is not None and prev[0] == fs) else 0
        ws = fs - k
        w0, s0, s1 = ws * hop / SAMPLE_RATE, fs * hop / SAMPLE_RATE, fe * hop / SAMPLE_RATE
        seg = src[ws * hop : fe * hop]
        if not any(s0 <= c["t0"] < s1 for c in chunks) or seg.numel() < hop:
            prev = None
            continue
        win_chunks = [dict(c, t0=c["t0"] - w0, t1=min(c["t1"], s1) - w0) for c in chunks if w0 <= c["t0"] < s1]
        ctx = prev[1][-k:] if k else None
        batch, _ = build_scene_batch(seg, win_chunks, lang, vae, spk, emo, stats, model.n_streams, device, ctx)
        x = model.sample(batch, steps=steps, guidance=guidance, sway=-1.0 if not step_cond else 0.0,
                         seed=seed + si, step_cond=step_cond)
        nf = int(batch["frame_mask"][0].sum())
        prev = (ws + nf, x[0, :nf])
        wav = vae.decode((x * std + mean).transpose(1, 2))[0].float().cpu()[: seg.numel()]
        if loudness:
            spans = [(int(c["t0"] * SAMPLE_RATE), int(c["t1"] * SAMPLE_RATE)) for c in win_chunks]
            wav = match_loudness(wav, seg, spans)
        room = None
        if acoustic:
            wav = match_eq(wav, None, SAMPLE_RATE, ref_bands=ltas_bands(seg, SAMPLE_RATE) or film_bands)
            room = fit_room(wav, decay_stats(seg, SAMPLE_RATE) or film_room, SAMPLE_RATE, seed=seed)
        own = wav[k * hop :].clone()  # the context part was already written by the previous scene
        fade = min(240, own.numel() // 4)
        if fade:
            ramp = torch.linspace(0, 1, fade)
            own[:fade] *= ramp
            own[-fade:] *= ramp.flip(0)
        if room is not None:
            own = apply_room(own, room["rt60"], room["wet"], SAMPLE_RATE, seed=seed, tail_sec=TAIL_SEC)
        a = fs * hop
        n = min(own.numel(), out.numel() - a)
        out[a : a + n] += own[:n]
    return out[: src.numel()]


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
    ap.add_argument("--no-acoustic-match", action="store_true", help="skip EQ / room matching to the original")
    ap.add_argument("--context-sec", type=float, default=None,
                    help="previous-scene context (s); default sample.context_sec, 0 disables")
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
    context_sec = args.context_sec if args.context_sec is not None else float(cfg.sample.get("context_sec", 0.0))
    out = dub(model, vae, stats, src, script, spk, emo, device, steps, guidance, max_sec, args.seed, step_cond,
              not args.no_loudness_match, not args.no_acoustic_match, context_sec)
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
