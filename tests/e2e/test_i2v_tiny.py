"""Spec 12: image-to-video on tiny models against the diffusers pipeline, float32, from the same noise."""

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.utils.compare import compare
from tests.parity.test_video_vae_encoder import synthetic_image

HEIGHT, WIDTH, NUM_FRAMES, STEPS = 32, 32, 9, 4


def run_both(tiny_pipelines, monkeypatch, image):
    torch, reference, pipeline, _ = tiny_pipelines
    from diffusers.models.autoencoders import vae as vae_module

    rng = np.random.default_rng(0)
    video_shape, audio_shape = pipeline.latent_shapes(HEIGHT, WIDTH, NUM_FRAMES, 24.0)
    inputs = {
        "latents": rng.standard_normal(video_shape).astype(np.float32),
        "audio_latents": rng.standard_normal(audio_shape).astype(np.float32),
        "prompt_embeds": rng.standard_normal((1, 5, 8)).astype(np.float32),
        "pooled": rng.standard_normal((1, 4)).astype(np.float32),
        "image_noise": rng.standard_normal(pipeline.image_noise_shape(HEIGHT, WIDTH)).astype(np.float32),
    }
    # The reference samples the image latent distribution with a torch generator; give it the same NumPy noise.
    noise = torch.from_numpy(inputs["image_noise"])
    monkeypatch.setattr(
        vae_module.DiagonalGaussianDistribution, "sample", lambda self, generator=None: self.mean + self.std * noise
    )
    steps = []

    def record(_pipe, _i, _t, tensors):
        steps.append((tensors["latents"].clone(), tensors["audio_latents"].clone()))
        return {}

    with torch.no_grad():
        ref_video, ref_audio = reference(
            image=image,
            prompt_embeds=torch.from_numpy(inputs["prompt_embeds"]),
            pooled_prompt_embeds=torch.from_numpy(inputs["pooled"]),
            latents=torch.from_numpy(inputs["latents"]),
            audio_latents=torch.from_numpy(inputs["audio_latents"]),
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=STEPS,
            guidance_scale=1.0,
            output_type="np",
            return_dict=False,
            callback_on_step_end=record,
            callback_on_step_end_tensor_inputs=["latents", "audio_latents"],
        )
    output = pipeline(
        {"encoder_hidden_states": mx.array(inputs["prompt_embeds"]), "pooled_projections": mx.array(inputs["pooled"])},
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        latents=inputs["latents"],
        audio_latents=inputs["audio_latents"],
        image=image,
        image_noise=inputs["image_noise"],
        return_step_latents=True,
    )
    return output, steps, ref_video, ref_audio


@pytest.mark.parametrize("size", [(48, 40), (32, 32)], ids=["crop", "exact"])
def test_i2v_matches_diffusers(tiny_pipelines, monkeypatch, size):
    output, steps, ref_video, ref_audio = run_both(tiny_pipelines, monkeypatch, synthetic_image(*size, seed=1))
    assert len(output.step_latents) == len(steps) == STEPS
    for i, ((video, audio), (ref_v, ref_a)) in enumerate(zip(output.step_latents, steps)):
        assert video.shape == tuple(ref_v.shape)  # includes the conditioning tail frame
        assert compare(video, ref_v)["rel_max"] <= 1e-4, i
        assert compare(audio, ref_a)["rel_max"] <= 1e-4, i
    assert output.frames.shape == ref_video.shape == (1, NUM_FRAMES, HEIGHT, WIDTH, 3)
    assert compare(output.frames, ref_video)["rel_max"] <= 1e-3
    assert compare(output.audio, ref_audio)["rel_max"] <= 1e-3


def test_image_changes_the_output(tiny_pipelines, monkeypatch):
    """Guards the test above: the image path really conditions (output differs from text-to-video)."""
    output, *_ = run_both(tiny_pipelines, monkeypatch, synthetic_image(32, 32, seed=1))
    _, _, pipeline, _ = tiny_pipelines
    rng = np.random.default_rng(0)
    video_shape, audio_shape = pipeline.latent_shapes(HEIGHT, WIDTH, NUM_FRAMES, 24.0)
    latents = rng.standard_normal(video_shape).astype(np.float32)
    audio = rng.standard_normal(audio_shape).astype(np.float32)
    text = {
        "encoder_hidden_states": mx.array(rng.standard_normal((1, 5, 8)).astype(np.float32)),
        "pooled_projections": mx.array(rng.standard_normal((1, 4)).astype(np.float32)),
    }
    t2v = pipeline(
        text,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        latents=latents,
        audio_latents=audio,
    )
    assert compare(output.frames, t2v.frames)["rel_max"] > 1e-3
