"""Shared tiny pipeline for the end-to-end tests."""

import json

import mlx.core as mx
import pytest

from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline


def build_tiny_pipelines(root, dit_overrides: dict | None = None, scheduler_name: str = "PiflowScheduler"):
    """A tiny diffusers pipeline, its components saved and converted under `root`, and the matching MLX pipeline."""
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

    components = {
        "transformer": build(Kandinsky6Transformer3DModel, tiny_configs.DIT | (dit_overrides or {})),
        "vae": build(AutoencoderKLHunyuanVideo, tiny_configs.VIDEO_VAE),
        # The DiT's audio latents have 40 channels; the audio VAE must match.
        "audio_vae": build(MMAudioVAE, tiny_configs.AUDIO_VAE | {"latent_channels": 40}),
        "vocoder": build(MMAudioVocoder, tiny_configs.VOCODER),
    }
    for name, model in components.items():
        model.save_pretrained(root / "ckpt" / name)
        convert_component(root / "ckpt" / name, name, root / "converted" / name)
    if scheduler_name == "PiflowScheduler":
        scheduler = PiflowScheduler(**tiny_configs.PIFLOW)
    else:
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    scheduler.save_pretrained(root / "ckpt" / "scheduler")
    (root / "converted" / "scheduler").mkdir(parents=True)
    config = dict(scheduler.config) | {"_class_name": scheduler_name}
    (root / "converted" / "scheduler" / "scheduler_config.json").write_text(json.dumps(config))
    reference = Kandinsky6TI2VAPipeline(
        text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None, scheduler=scheduler, **components
    )
    return torch, reference, Kandinsky6Pipeline.from_converted(root / "converted", dtype=mx.float32), root


@pytest.fixture(scope="session")
def tiny_pipelines(tmp_path_factory):
    """Distilled-style tiny pipeline: PiFlow, GM outputs (out dims 10x the inputs)."""
    return build_tiny_pipelines(tmp_path_factory.mktemp("tiny_pipeline"))


@pytest.fixture(scope="session")
def tiny_base_pipelines(tmp_path_factory):
    """Non-distilled-style tiny pipeline (spec 13): FlowMatch Euler, velocity outputs (out dims = in dims)."""
    return build_tiny_pipelines(
        tmp_path_factory.mktemp("tiny_base_pipeline"),
        {"out_visual_dim": 16, "out_audio_dim": 40},
        "FlowMatchEulerDiscreteScheduler",
    )
