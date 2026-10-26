"""DubFlow: time-synchronous flow-matching model for scene-level dubbing."""

__version__ = "0.1.0"

SAMPLE_RATE = 24000
LATENT_HOP = 960  # 24000 / 960 = 25 Hz latent frame rate
FRAME_RATE = SAMPLE_RATE / LATENT_HOP
