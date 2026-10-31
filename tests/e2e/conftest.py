"""Shared tiny pipeline for the end-to-end tests."""

import json

import mlx.core as mx
import pytest

from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline


@pytest.fixture(scope="session")
def tiny_pipelines(tmp_path_factory):
    """A tiny diffusers pipeline, its components saved and converted, and the matching MLX pipeline."""
    torch = pytest.importorskip("torch")
    from diffusers import (
        AutoencoderKLHunyuanVideo,
        Kandinsky6TI2VAPipeline,
        Kandinsky6Transformer3DModel,
        PiflowScheduler,
    )
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    from kandinsky6_mlx.convert.convert import convert_component
    from reference import tiny_configs
    from reference.make_tiny_fixtures import build

    root = tmp_path_factory.mktemp("tiny_pipeline")
    components = {
        "transformer": build(Kandinsky6Transformer3DModel, tiny_configs.DIT),
        "vae": build(AutoencoderKLHunyuanVideo, tiny_configs.VIDEO_VAE),
        # The DiT's audio latents have 40 channels; the audio VAE must match.
        "audio_vae": build(MMAudioVAE, tiny_configs.AUDIO_VAE | {"latent_channels": 40}),
        "vocoder": build(MMAudioVocoder, tiny_configs.VOCODER),
    }
    for name, model in components.items():
        model.save_pretrained(root / "ckpt" / name)
        convert_component(root / "ckpt" / name, name, root / "converted" / name)
    scheduler = PiflowScheduler(**tiny_configs.PIFLOW)
    scheduler.save_pretrained(root / "ckpt" / "scheduler")
    (root / "converted" / "scheduler").mkdir(parents=True)
    (root / "converted" / "scheduler" / "scheduler_config.json").write_text(json.dumps(dict(scheduler.config)))
    reference = Kandinsky6TI2VAPipeline(
        text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, scheduler=scheduler, **components
    )
    return torch, reference, Kandinsky6Pipeline.from_converted(root / "converted", dtype=mx.float32), root
