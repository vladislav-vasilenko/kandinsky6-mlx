"""Spec 13: FlowMatch Euler + classifier-free guidance on tiny models against the diffusers pipeline, float32."""

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.pipelines.t2v import DEFAULT_NEGATIVE_PROMPT
from kandinsky6_mlx.utils.compare import compare
from tests.parity.test_video_vae_encoder import synthetic_image

HEIGHT, WIDTH, NUM_FRAMES, STEPS, GUIDANCE = 32, 32, 9, 6, 5.0


def inputs(pipeline):
    rng = np.random.default_rng(0)
    video_shape, audio_shape = pipeline.latent_shapes(HEIGHT, WIDTH, NUM_FRAMES, 24.0)
    return {
        "latents": rng.standard_normal(video_shape).astype(np.float32),
        "audio_latents": rng.standard_normal(audio_shape).astype(np.float32),
        "prompt_embeds": rng.standard_normal((1, 5, 8)).astype(np.float32),
        "pooled": rng.standard_normal((1, 4)).astype(np.float32),
        "negative_embeds": rng.standard_normal((1, 3, 8)).astype(np.float32),  # a different length on purpose
        "negative_pooled": rng.standard_normal((1, 4)).astype(np.float32),
        "image_noise": rng.standard_normal(pipeline.image_noise_shape(HEIGHT, WIDTH)).astype(np.float32),
    }


def run_reference(torch, reference, x, image=None, monkeypatch=None):
    steps = []

    def record(_pipe, _i, _t, tensors):
        steps.append(tensors["latents"].clone())
        return {}

    if image is not None:
        from diffusers.models.autoencoders import vae as vae_module

        noise = torch.from_numpy(x["image_noise"])
        monkeypatch.setattr(
            vae_module.DiagonalGaussianDistribution, "sample", lambda self, generator=None: self.mean + self.std * noise
        )
    with torch.no_grad():
        video, audio = reference(
            image=image,
            prompt_embeds=torch.from_numpy(x["prompt_embeds"]),
            pooled_prompt_embeds=torch.from_numpy(x["pooled"]),
            negative_prompt_embeds=torch.from_numpy(x["negative_embeds"]),
            negative_pooled_prompt_embeds=torch.from_numpy(x["negative_pooled"]),
            latents=torch.from_numpy(x["latents"]),
            audio_latents=torch.from_numpy(x["audio_latents"]),
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            output_type="np",
            return_dict=False,
            callback_on_step_end=record,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    return steps, video, audio


def run_mlx(pipeline, x, batched, image=None):
    return pipeline(
        {"encoder_hidden_states": mx.array(x["prompt_embeds"]), "pooled_projections": mx.array(x["pooled"])},
        negative_text_embeds={
            "encoder_hidden_states": mx.array(x["negative_embeds"]),
            "pooled_projections": mx.array(x["negative_pooled"]),
        },
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        latents=x["latents"],
        audio_latents=x["audio_latents"],
        cfg_batched=batched,
        image=image,
        image_noise=x["image_noise"] if image is not None else None,
        return_step_latents=True,
    )


@pytest.mark.parametrize("batched", [False, True], ids=["sequential", "batched"])
@pytest.mark.parametrize("i2v", [False, True], ids=["t2v", "i2v"])
def test_cfg_matches_diffusers(tiny_base_pipelines, monkeypatch, batched, i2v):
    torch, reference, pipeline, _ = tiny_base_pipelines
    x = inputs(pipeline)
    image = synthetic_image(48, 40, seed=3) if i2v else None
    steps, ref_video, ref_audio = run_reference(torch, reference, x, image, monkeypatch)
    output = run_mlx(pipeline, x, batched, image)
    assert len(output.step_latents) == len(steps) == STEPS
    for i, ((video, _), ref) in enumerate(zip(output.step_latents, steps)):
        assert compare(video, ref)["rel_max"] <= 1e-4, i
    assert compare(output.frames, ref_video)["rel_max"] <= 1e-3
    assert compare(output.audio, ref_audio)["rel_max"] <= 1e-3


def test_batched_equals_sequential(tiny_base_pipelines):
    _, _, pipeline, _ = tiny_base_pipelines
    x = inputs(pipeline)
    a, b = run_mlx(pipeline, x, False), run_mlx(pipeline, x, True)
    for (va, _), (vb, _) in zip(a.step_latents, b.step_latents):
        assert compare(va, vb)["rel_max"] <= 1e-5


def test_guidance_changes_the_output(tiny_base_pipelines):
    """Guards the tests above: guidance 5 differs from guidance 1 on the same noise."""
    _, _, pipeline, _ = tiny_base_pipelines
    x = inputs(pipeline)
    guided = run_mlx(pipeline, x, False)
    plain = pipeline(
        {"encoder_hidden_states": mx.array(x["prompt_embeds"]), "pooled_projections": mx.array(x["pooled"])},
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
        latents=x["latents"],
        audio_latents=x["audio_latents"],
    )
    assert compare(guided.frames, plain.frames)["rel_max"] > 1e-3


def test_scheduler_defaults_follow_the_checkpoint(tiny_base_pipelines, tiny_pipelines):
    """Non-distilled: FlowMatch, 50 steps, guidance 5 needs negative embeddings; distilled stays PiFlow, 10, 1."""
    _, _, base, _ = tiny_base_pipelines
    x = inputs(base)
    with pytest.raises(ValueError, match="negative_text_embeds"):
        base(
            {"encoder_hidden_states": mx.array(x["prompt_embeds"]), "pooled_projections": mx.array(x["pooled"])},
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
        )


def test_default_negative_prompt_matches_diffusers():
    pytest.importorskip("diffusers")
    from diffusers import Kandinsky6TI2VAPipeline

    assert DEFAULT_NEGATIVE_PROMPT == Kandinsky6TI2VAPipeline._DEFAULT_NEGATIVE_PROMPT


def test_stage_b_harness_with_guidance(tiny_base_pipelines, tmp_path):
    """record_pipeline_run.py with FlowMatch, guidance 5 and negative embeddings on the tiny non-distilled snapshot,
    then the MLX side in float32 through the stage B harness."""
    from kandinsky6_mlx.utils.fixtures import save_fixture
    from reference import record_pipeline_run
    from tests.e2e.test_pipeline_real import stage_b_metrics

    torch, _, _, root = tiny_base_pipelines
    g = torch.Generator().manual_seed(0)
    for name, n in (("text", 6), ("negative", 4)):
        save_fixture(
            tmp_path / name,
            {
                "inputs.encoder_hidden_states": torch.randn(1, n, 8, generator=g),
                "inputs.pooled_projections": torch.randn(1, 4, generator=g),
            },
            {"fixture": name},
        )
    out = tmp_path / "run"
    args = [
        "--checkpoint",
        str(root / "ckpt"),
        "--text-embeds",
        str(tmp_path / "text"),
        "--steps",
        "4",
        "--guidance",
        "5",
        "--negative-text-embeds",
        str(tmp_path / "negative"),
        "--out",
        str(out),
    ]
    assert record_pipeline_run.main(args) == 0
    result = stage_b_metrics(root / "converted", out, out, dtype=mx.float32)
    assert len(result["steps"]) == 4
    assert result["frames_psnr_db"] > 60 and result["audio_snr_db"] > 60
    for entry in result["steps"]:
        for m in entry.values():
            assert m["cosine"] > 0.99999 and m["rel_l2"] < 1e-4
