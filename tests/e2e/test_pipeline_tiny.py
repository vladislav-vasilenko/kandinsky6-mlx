"""Spec 09: the MLX text-to-video+audio pipeline against the diffusers pipeline on tiny models, float32."""

import math

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.pipelines.export import write_mp4
from kandinsky6_mlx.pipelines.t2v import Kandinsky6Pipeline, load_text_embeds
from kandinsky6_mlx.utils.compare import compare

HEIGHT, WIDTH, NUM_FRAMES, STEPS = 32, 32, 9, 4


def _inputs(pipeline: Kandinsky6Pipeline):
    rng = np.random.default_rng(0)
    video_shape, audio_shape = pipeline.latent_shapes(HEIGHT, WIDTH, NUM_FRAMES, 24.0)
    return {
        "latents": rng.standard_normal(video_shape).astype(np.float32),
        "audio_latents": rng.standard_normal(audio_shape).astype(np.float32),
        "prompt_embeds": rng.standard_normal((1, 5, 8)).astype(np.float32),
        "pooled": rng.standard_normal((1, 4)).astype(np.float32),
    }


def test_latent_shapes_follow_the_reference(tiny_pipelines):
    _, reference, pipeline, _ = tiny_pipelines
    video, audio = pipeline.latent_shapes(256, 432, 25, 24.0)
    assert video == (1, 16, 7, 32, 54)
    # Released hop 1024 gives 45 latent frames for Smoke; the tiny audio VAE's hop is 32.
    assert pipeline.audio_latent_hop_length == reference.audio_latent_hop_length
    assert audio == (1, 40, math.ceil(25 / 24 * 44100 / reference.audio_latent_hop_length))
    # The reference's num_frames // 4 * 4 + 1.
    assert [pipeline.round_num_frames(n) for n in (23, 24, 25, 26)] == [21, 25, 25, 25]


def test_pipeline_matches_diffusers_from_the_same_noise(tiny_pipelines, tmp_path):
    torch, reference, pipeline, _ = tiny_pipelines
    inputs = _inputs(pipeline)
    steps = []

    def record(_pipe, _i, _t, tensors):
        steps.append((tensors["latents"].clone(), tensors["audio_latents"].clone()))
        return {}

    with torch.no_grad():
        ref_video, ref_audio = reference(
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
        return_step_latents=True,
    )
    assert len(output.step_latents) == len(steps) == STEPS
    for i, ((video, audio), (ref_v, ref_a)) in enumerate(zip(output.step_latents, steps)):
        assert compare(video, ref_v)["rel_max"] <= 1e-4, i
        assert compare(audio, ref_a)["rel_max"] <= 1e-4, i
    assert output.frames.shape == ref_video.shape == (1, NUM_FRAMES, HEIGHT, WIDTH, 3)
    assert compare(output.frames, ref_video)["rel_max"] <= 1e-3
    assert output.audio.shape == ref_audio.shape
    assert compare(output.audio, ref_audio)["rel_max"] <= 1e-3
    assert set(output.timings) == set(output.peak_memory_gib) == {"denoise", "video_decode", "audio_decode"}

    path = write_mp4(tmp_path / "tiny.mp4", output.frames[0], output.frame_rate, output.audio[0], output.sample_rate)
    import av

    with av.open(str(path)) as container:
        kinds = {s.type for s in container.streams}
        decoded = sum(1 for _ in container.decode(video=0))
    assert kinds == {"video", "audio"} and decoded == NUM_FRAMES


def test_text_embeds_file_trims_padding(tmp_path):
    from safetensors.numpy import save_file

    embeds = np.random.default_rng(0).standard_normal((1, 6, 8)).astype(np.float32)
    mask = np.array([[1, 1, 1, 1, 0, 0]], bool)
    save_file(
        {
            "inputs.encoder_hidden_states": embeds,
            "inputs.pooled_projections": np.zeros((1, 4), np.float32),
            "inputs.encoder_attention_mask": mask,
        },
        str(tmp_path / "text.safetensors"),
    )
    trimmed = load_text_embeds(tmp_path / "text")
    assert trimmed["encoder_hidden_states"].shape == (1, 4, 8) and "encoder_attention_mask" not in trimmed
    kept = load_text_embeds(tmp_path / "text", trim_padding=False)
    assert kept["encoder_hidden_states"].shape == (1, 6, 8) and "encoder_attention_mask" in kept


def test_guidance_is_not_supported_yet(tiny_pipelines):
    _, _, pipeline, _ = tiny_pipelines
    with pytest.raises(NotImplementedError):
        pipeline({}, guidance_scale=5.0)
