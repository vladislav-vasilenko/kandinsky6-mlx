"""Build tiny random-weight reference models and save their weights, inputs, outputs and submodule activations.

Usage:
    uv run --group reference python reference/make_tiny_fixtures.py --out fixtures/tiny

Weights come from `torch.manual_seed(0)` plus deterministic noise on parameters that the default initialization
leaves constant (norm scales and biases, gains, Snake parameters), so that every parameter affects the output.
Inputs come from NumPy (`np.random.default_rng`). Two runs write byte-identical safetensors files.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLHunyuanVideo, Kandinsky6Transformer3DModel, PiflowScheduler
from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.utils.fixtures import save_fixture
from reference import tiny_configs
from reference.recorder import ActivationRecorder, flatten, library_versions, modules_of_classes

DIT_CLASSES = [
    "Kandinsky6RoPE1D",
    "Kandinsky6RoPE3D",
    "Kandinsky6TimeEmbeddings",
    "Kandinsky6TextEmbeddings",
    "Kandinsky6VisualEmbeddings",
    "Kandinsky6Modulation",
    "Kandinsky6Attention",
    "FeedForward",
    "Embedding",
    "Kandinsky6TransformerEncoderBlock",
    "Kandinsky6FusedTransformerDecoderBlock",
    "Kandinsky6OutLayer",
    "Kandinsky6OutLayerAudio",
]
VIDEO_VAE_CLASSES = [
    "HunyuanVideoResnetBlockCausal3D",
    "HunyuanVideoMidBlock3D",
    "HunyuanVideoUpBlock3D",
    "HunyuanVideoUpsampleCausal3D",
    "Attention",
    "HunyuanVideoDecoder3D",
]
AUDIO_VAE_CLASSES = ["MMAudioResnetBlock1D", "MMAudioAttnBlock1D", "MMAudioUpsample1D", "MMAudioDecoder1D"]
VOCODER_CLASSES = [
    "MMAudioSnakeBeta",
    "MMAudioUpSample1d",
    "MMAudioDownSample1d",
    "MMAudioActivation1d",
    "MMAudioAMPBlock",
    "ConvTranspose1d",
]


def build(cls, config: dict, extra_buffers: tuple[str, ...] = ()) -> torch.nn.Module:
    torch.manual_seed(0)
    model = cls(**config).float().eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            flat = param.flatten()
            if not bool((flat == flat[0]).all()):
                continue
            noise = 0.1 * torch.randn(param.shape, generator=generator)
            base = 1.0 if (name.endswith("weight") and float(flat[0]) == 1.0) else 0.0
            param.copy_(base + noise)
        buffers = dict(model.named_buffers())
        for name in extra_buffers:
            buffers[name].copy_(buffers[name] + 0.1 * torch.randn(buffers[name].shape, generator=generator).abs())
    return model


def rng_array(rng: np.random.Generator, shape, scale: float = 1.0) -> torch.Tensor:
    return torch.from_numpy((scale * rng.standard_normal(shape)).astype(np.float32))


def weights_of(model: torch.nn.Module, prefix: str = "weights") -> dict[str, torch.Tensor]:
    return {f"{prefix}.{k}": v.detach().clone().contiguous() for k, v in model.state_dict().items()}


def meta_for(name: str, config: dict, reference: str, seeds: dict, calls: dict, **extra) -> dict:
    return {
        "fixture": name,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
        "reference": reference,
        "seeds": seeds,
        "versions": library_versions(),
        "calls": calls,
        **extra,
    }


def dit_inputs(rng: np.random.Generator, scenario: str) -> dict:
    cfg = tiny_configs.DIT
    num_frames = 3
    channels = 2 * cfg["in_visual_dim"] + 1
    inputs = {
        "hidden_states": rng_array(rng, (1, num_frames, 4, 6, channels)),
        "encoder_hidden_states": rng_array(rng, (1, 5, cfg["in_text_dim"])),
        "pooled_projections": rng_array(rng, (1, cfg["in_text_dim2"])),
        "timestep": torch.tensor([437.5]),
    }
    if scenario in ("av", "rope_pos"):
        inputs["audio_hidden_states"] = rng_array(rng, (1, 6, cfg["in_audio_dim"]))
    if scenario == "av":
        inputs["encoder_attention_mask"] = torch.tensor([[True, True, True, True, False]])
        inputs["visual_token_type_ids"] = torch.tensor([[1, 0, 0]])
    if scenario == "rope_pos":
        # I2V appends a reference frame that reuses temporal position 0.
        inputs["visual_rope_pos"] = (torch.tensor([0, 1, 0]), torch.arange(2), torch.arange(3))
        inputs["visual_token_type_ids"] = torch.tensor([[0, 0, 1]])
    return inputs


def run_dit(model, inputs: dict, record: list[str], tag: str, tensors: dict, calls: dict) -> None:
    for key, value in inputs.items():
        flatten(value, f"inputs.{tag}.{key}", tensors)
    with ActivationRecorder(model, record, tag=tag) as recorder, torch.no_grad():
        output = model(**inputs, return_dict=False)
    tensors.update(recorder.tensors)
    calls[tag] = recorder.calls
    tensors[f"outputs.{tag}.sample"] = output[0].clone()
    if len(output) > 1:
        tensors[f"outputs.{tag}.audio_sample"] = output[1].clone()


def make_dit(out: Path) -> None:
    model = build(Kandinsky6Transformer3DModel, tiny_configs.DIT)
    all_modules = modules_of_classes(model, DIT_CLASSES)
    blocks = modules_of_classes(model, ["Kandinsky6TransformerEncoderBlock", "Kandinsky6FusedTransformerDecoderBlock"])
    tensors, calls = weights_of(model), {}
    for seed, scenario in enumerate(("av", "video_only", "rope_pos")):
        rng = np.random.default_rng(seed)
        record = all_modules if scenario == "av" else blocks
        run_dit(model, dit_inputs(rng, scenario), record, scenario, tensors, calls)
    save_fixture(
        out / "dit",
        tensors,
        meta_for(
            "dit",
            tiny_configs.DIT,
            "diffusers.Kandinsky6Transformer3DModel.forward",
            {"torch_manual_seed": 0, "param_noise_generator": 1, "numpy": {"av": 0, "video_only": 1, "rope_pos": 2}},
            calls,
            scenarios={
                "av": "video+audio, padded text mask, token types; every primitive and block recorded",
                "video_only": "no audio; blocks recorded",
                "rope_pos": "custom visual_rope_pos with repeated temporal 0; blocks recorded",
            },
        ),
    )


def make_dit_variants(out: Path) -> None:
    """Fused-block flag combinations other than the released one, for branch coverage."""
    tensors, calls, variants = {}, {}, {}
    blocks_cls = ["Kandinsky6FusedTransformerDecoderBlock"]
    for cross_gates, fix_modulation, ca_rope in itertools.product((True, False), repeat=3):
        if cross_gates and fix_modulation and ca_rope:
            continue
        tag = f"cg{int(cross_gates)}_fm{int(fix_modulation)}_ca{int(ca_rope)}"
        config = dict(tiny_configs.DIT, cross_gates=cross_gates, fix_modulation=fix_modulation, ca_rope=ca_rope)
        variants[tag] = {"cross_gates": cross_gates, "fix_modulation": fix_modulation, "ca_rope": ca_rope}
        model = build(Kandinsky6Transformer3DModel, config)
        tensors.update(weights_of(model, f"weights.{tag}"))
        run_dit(
            model,
            dit_inputs(np.random.default_rng(0), "av"),
            modules_of_classes(model, blocks_cls),
            tag,
            tensors,
            calls,
        )
    save_fixture(
        out / "dit_variants",
        tensors,
        meta_for(
            "dit_variants",
            tiny_configs.DIT,
            "diffusers.Kandinsky6Transformer3DModel.forward",
            {"torch_manual_seed": 0, "param_noise_generator": 1, "numpy": 0},
            calls,
            variants=variants,
        ),
    )


def make_video_vae(out: Path) -> None:
    model = build(AutoencoderKLHunyuanVideo, tiny_configs.VIDEO_VAE)
    rng = np.random.default_rng(0)
    latents = rng_array(rng, (1, 16, 2, 4, 4))
    record = ["post_quant_conv", "decoder.conv_in", "decoder.conv_out"] + [
        name for name in modules_of_classes(model, VIDEO_VAE_CLASSES) if name.startswith("decoder")
    ]
    tensors, calls = weights_of(model), {}
    tensors["inputs.decode.z"] = latents
    with ActivationRecorder(model, record, tag="decode") as recorder, torch.no_grad():
        sample = model.decode(latents).sample
    tensors.update(recorder.tensors)
    calls["decode"] = recorder.calls
    tensors["outputs.decode.sample"] = sample
    save_fixture(
        out / "video_vae_decoder",
        tensors,
        meta_for(
            "video_vae_decoder",
            tiny_configs.VIDEO_VAE,
            "diffusers.AutoencoderKLHunyuanVideo.decode",
            {"torch_manual_seed": 0, "param_noise_generator": 1, "numpy": 0},
            calls,
        ),
    )


def make_audio_vae(out: Path) -> None:
    model = build(MMAudioVAE, tiny_configs.AUDIO_VAE, extra_buffers=("vae.data_mean", "vae.data_std"))
    rng = np.random.default_rng(0)
    latents = rng_array(rng, (1, tiny_configs.AUDIO_VAE["latent_channels"], 6))
    record = modules_of_classes(model.vae.decoder, AUDIO_VAE_CLASSES)
    record = ["vae.decoder"] + [f"vae.decoder.{name}" for name in record] + ["vae.decoder.conv_in"]
    tensors, calls = weights_of(model), {}
    tensors["inputs.decode.z"] = latents
    with ActivationRecorder(model, record, tag="decode") as recorder, torch.no_grad():
        mel = model.decode(latents).sample
    tensors.update(recorder.tensors)
    calls["decode"] = recorder.calls
    tensors["outputs.decode.sample"] = mel
    save_fixture(
        out / "audio_vae_decoder",
        tensors,
        meta_for(
            "audio_vae_decoder",
            tiny_configs.AUDIO_VAE,
            "diffusers.MMAudioVAE.decode",
            {"torch_manual_seed": 0, "param_noise_generator": 1, "numpy": 0},
            calls,
        ),
    )


def make_vocoder(out: Path) -> None:
    model = build(MMAudioVocoder, tiny_configs.VOCODER)
    rng = np.random.default_rng(0)
    mel = rng_array(rng, (1, tiny_configs.VOCODER["num_mels"], 12))
    tensors, calls = weights_of(model), {}
    tensors["inputs.forward.mel"] = mel
    record = modules_of_classes(model, VOCODER_CLASSES)
    with ActivationRecorder(model, record, tag="forward") as recorder, torch.no_grad():
        waveform = model(mel).sample
    tensors.update(recorder.tensors)
    calls["forward"] = recorder.calls
    tensors["outputs.forward.sample"] = waveform
    save_fixture(
        out / "vocoder",
        tensors,
        meta_for(
            "vocoder",
            tiny_configs.VOCODER,
            "diffusers.MMAudioVocoder.forward",
            {"torch_manual_seed": 0, "param_noise_generator": 1, "numpy": 0},
            calls,
        ),
    )


def make_piflow(out: Path, num_steps: int = 10) -> None:
    config = tiny_configs.PIFLOW
    n_grid = config["n_grid"]
    shapes = {"video": (1, 3, 2, 3, 16), "audio": (1, 6, 40)}
    tensors = {}
    for seed, (stream, shape) in enumerate(shapes.items()):
        scheduler = PiflowScheduler(**config)
        scheduler.set_timesteps(num_steps)
        rng = np.random.default_rng(seed)
        sample = rng_array(rng, shape)
        tensors[f"outputs.{stream}.timesteps"] = scheduler.timesteps.clone()
        tensors[f"outputs.{stream}.sigmas"] = scheduler.sigmas.clone()
        tensors[f"inputs.{stream}.sample"] = sample
        for step, timestep in enumerate(scheduler.timesteps):
            model_output = rng_array(rng, (*shape[:-1], shape[-1] * n_grid))
            tensors[f"inputs.{stream}.model_output.{step}"] = model_output
            sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
            tensors[f"outputs.{stream}.sample.{step}"] = sample.clone()
    save_fixture(
        out / "piflow",
        tensors,
        meta_for(
            "piflow",
            config,
            "diffusers.PiflowScheduler.set_timesteps + step",
            {"numpy": {"video": 0, "audio": 1}},
            {},
            num_inference_steps=num_steps,
            shapes={k: list(v) for k, v in shapes.items()},
            note="model_output.<i> are random stand-ins for the widened DiT output at step i",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("fixtures/tiny"))
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    for make in (make_dit, make_dit_variants, make_video_vae, make_audio_vae, make_vocoder, make_piflow):
        make(args.out)
        print(f"{make.__name__}: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
