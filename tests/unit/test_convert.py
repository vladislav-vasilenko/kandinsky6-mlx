"""Weight conversion: names, MLX layouts, dtype policy, dropped keys and portability."""

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.convert.convert import convert_checkpoint, convert_component, read_header, read_tensor
from kandinsky6_mlx.convert.layouts import conv_rule
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.dtype_policy import DtypePolicy, keeps_fp32

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
from torch import nn

from reference import tiny_configs
from reference.make_tiny_fixtures import build

ROOT = Path(__file__).resolve().parents[2]
CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d)


def classes():
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    return {
        "transformer": (diffusers.Kandinsky6Transformer3DModel, tiny_configs.DIT),
        "vae": (diffusers.AutoencoderKLHunyuanVideo, tiny_configs.VIDEO_VAE),
        "audio_vae": (MMAudioVAE, tiny_configs.AUDIO_VAE),
        "vocoder": (MMAudioVocoder, tiny_configs.VOCODER),
    }


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("checkpoint")
    models = {}
    for component, (cls, config) in classes().items():
        model = build(cls, config)
        model.save_pretrained(root / component)
        models[component] = model
    return root, models


@pytest.fixture(scope="module")
def converted(tiny_checkpoint, tmp_path_factory):
    root, _ = tiny_checkpoint
    out = tmp_path_factory.mktemp("converted")
    reports = convert_checkpoint(root, out)
    return out, reports


def load_converted(out: Path, component: str) -> dict:
    return mx.load(str(out / component / "model.safetensors"))


def conv_weights(model) -> dict:
    return {f"{name}.weight": module for name, module in model.named_modules() if isinstance(module, CONV_TYPES)}


@pytest.mark.parametrize("component", ["transformer", "vae", "audio_vae", "vocoder"])
def test_names_layouts_and_values(component, tiny_checkpoint, converted):
    _, models = tiny_checkpoint
    out, reports = converted
    model = models[component]
    state = model.state_dict()
    weights = load_converted(out, component)
    assert set(weights) == set(state)
    convs = conv_weights(model)
    assert set(reports[component]["transposed_keys"]) == set(convs)
    for name, tensor in state.items():
        expected = tensor.numpy()
        rule = conv_rule(component, name)
        if rule is not None:
            expected = expected.transpose(rule.axes())
        actual = np.array(weights[name])
        assert actual.shape == expected.shape, name
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    assert json.loads((out / component / "config.json").read_text()) == json.loads(
        (tiny_checkpoint[0] / component / "config.json").read_text()
    )


def test_conv_layout_table_has_no_false_positives(tiny_checkpoint):
    _, models = tiny_checkpoint
    for component, model in models.items():
        matched = {name for name in model.state_dict() if conv_rule(component, name)}
        assert matched == set(conv_weights(model)), component


def test_mlx_convolutions_match_torch(tiny_checkpoint, converted):
    _, models = tiny_checkpoint
    out, _ = converted
    rng = np.random.default_rng(0)

    # Conv3d: the causal wrapper pads outside the inner conv, so the inner conv has no padding.
    conv3d = models["vae"].decoder.conv_in.conv
    x = rng.standard_normal((1, conv3d.in_channels, 3, 6, 5)).astype(np.float32)
    w = load_converted(out, "vae")["decoder.conv_in.conv.weight"]
    b = load_converted(out, "vae")["decoder.conv_in.conv.bias"]
    ref = conv3d(torch.from_numpy(x)).detach().numpy()
    got = mx.conv3d(mx.array(x.transpose(0, 2, 3, 4, 1)), w, padding=conv3d.padding) + b
    assert compare(np.array(got).transpose(0, 4, 1, 2, 3), ref)["rel_max"] <= 1e-5

    # Conv1d (magnitude-preserving conv, padding k // 2, no bias).
    conv1d = models["audio_vae"].vae.decoder.conv_in
    x = rng.standard_normal((1, conv1d.in_channels, 7)).astype(np.float32)
    w = load_converted(out, "audio_vae")["vae.decoder.conv_in.weight"]
    ref = conv1d(torch.from_numpy(x)).detach().numpy()
    got = mx.conv1d(mx.array(x.transpose(0, 2, 1)), w, padding=conv1d.padding[0])
    assert compare(np.array(got).transpose(0, 2, 1), ref)["rel_max"] <= 1e-5

    # ConvTranspose1d (vocoder upsampling).
    up = models["vocoder"].ups[0][0]
    x = rng.standard_normal((1, up.in_channels, 5)).astype(np.float32)
    weights = load_converted(out, "vocoder")
    ref = up(torch.from_numpy(x)).detach().numpy()
    got = (
        mx.conv_transpose1d(
            mx.array(x.transpose(0, 2, 1)), weights["ups.0.0.weight"], stride=up.stride[0], padding=up.padding[0]
        )
        + weights["ups.0.0.bias"]
    )
    assert compare(np.array(got).transpose(0, 2, 1), ref)["rel_max"] <= 1e-5


def test_bf16_data_is_byte_identical(tmp_path):
    from safetensors.torch import save_file

    src = tmp_path / "src" / "transformer"
    src.mkdir(parents=True)
    tensors = {
        "blocks.0.attn.to_query.weight": torch.randn(6, 4).to(torch.bfloat16),
        "video_time_embeddings.linear_1.weight": torch.randn(3, 2).to(torch.float32),
        "visual_modulation.out_layer.weight": torch.randn(5, 2).to(torch.bfloat16),
    }
    save_file(tensors, str(src / "diffusion_pytorch_model.safetensors"))
    report = convert_component(src, "transformer", tmp_path / "out")
    out = tmp_path / "out" / "model.safetensors"
    header, start = read_header(out)
    for name, tensor in tensors.items():
        stored = "BF16" if tensor.dtype == torch.bfloat16 else "F32"
        assert header[name]["dtype"] == stored, name
        raw = read_tensor(out, header[name], start)
        expected = tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor
        assert raw.tobytes() == expected.numpy().tobytes(), name
    assert report["dtype_counts"] == {"BF16": 2, "F32": 1}


def test_dtype_policy_on_bf16_dit(tmp_path):
    model = build(diffusers.Kandinsky6Transformer3DModel, tiny_configs.DIT).to(torch.bfloat16)
    model.save_pretrained(tmp_path / "ckpt" / "transformer")
    stored, _ = read_header(tmp_path / "ckpt" / "transformer" / "diffusion_pytorch_model.safetensors")
    convert_component(tmp_path / "ckpt" / "transformer", "transformer", tmp_path / "bf16")
    header, _ = read_header(tmp_path / "bf16" / "model.safetensors")
    assert {name: info["dtype"] for name, info in header.items()} == {
        name: info["dtype"] for name, info in stored.items()
    }
    assert keeps_fp32("transformer", "visual_transformer_blocks.0.video_dec_block.visual_modulation.out_layer.weight")
    assert keeps_fp32("transformer", "video_time_embeddings.linear_1.weight")
    assert not keeps_fp32("transformer", "visual_transformer_blocks.0.video_dec_block.self_attention.to_query.weight")
    assert not keeps_fp32("vae", "decoder.mid_block.attentions.0.to_q.weight")

    convert_component(
        tmp_path / "ckpt" / "transformer", "transformer", tmp_path / "fp32", dtype_policy=DtypePolicy("float32")
    )
    header32, _ = read_header(tmp_path / "fp32" / "model.safetensors")
    assert {info["dtype"] for info in header32.values()} == {"F32"}
    weights = mx.load(str(tmp_path / "fp32" / "model.safetensors"))
    for name, tensor in model.state_dict().items():
        np.testing.assert_array_equal(np.array(weights[name]), tensor.float().numpy(), err_msg=name)


def test_audio_vae_drops_embedded_vocoder(tmp_path, tiny_checkpoint):
    from safetensors.torch import load_file, save_file

    root, models = tiny_checkpoint
    src = tmp_path / "audio_vae"
    src.mkdir()
    tensors = load_file(str(root / "audio_vae" / "diffusion_pytorch_model.safetensors"))
    vocoder = {f"vocoder.{k}": v for k, v in models["vocoder"].state_dict().items()}
    save_file({**tensors, **vocoder}, str(src / "diffusion_pytorch_model.safetensors"))
    report = convert_component(src, "audio_vae", tmp_path / "out")
    weights = mx.load(str(tmp_path / "out" / "model.safetensors"))
    assert set(weights) == set(tensors)
    assert report["dropped_keys"] == len(vocoder)
    assert report["dropped_prefixes"] == ["vocoder."]


def test_converter_runs_without_mlx_and_torch(tmp_path, tiny_checkpoint):
    root, _ = tiny_checkpoint
    code = (
        "import sys; sys.modules['mlx'] = None; sys.modules['mlx.core'] = None; sys.modules['torch'] = None\n"
        "from kandinsky6_mlx.convert.convert import main\n"
        f"raise SystemExit(main(['--checkpoint', {str(root)!r}, '--out', {str(tmp_path / 'out')!r}]))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stderr
    for component in ("transformer", "vae", "audio_vae", "vocoder"):
        assert (tmp_path / "out" / component / "conversion.json").is_file()


def test_revision_from_manifest(tmp_path, tiny_checkpoint):
    import shutil

    root, _ = tiny_checkpoint
    ckpt = tmp_path / "ckpt"
    shutil.copytree(root / "vocoder", ckpt / "vocoder")
    (ckpt / "weights-manifest.json").write_text(json.dumps({"revision": "4c487e8b" + "0" * 32}))
    reports = convert_checkpoint(ckpt, tmp_path / "out", components=("vocoder",))
    assert reports["vocoder"]["source_revision"] == "4c487e8b" + "0" * 32
