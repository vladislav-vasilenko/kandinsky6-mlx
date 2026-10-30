"""Spec 05 stage A: the full MLX DiT against the diffusers 0.42.1 reference on the tiny fixture and the real configs."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from kandinsky6_mlx.models.dit.model import Kandinsky6Transformer3DModel, parameter_shapes
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.dtype_policy import keeps_fp32
from kandinsky6_mlx.utils.fixtures import load_fixture, load_module_weights, select, unflatten
from tests.parity.dit_cases import TINY

CONFIGS = Path(__file__).resolve().parents[2] / "fixtures" / "configs"
SCENARIOS = ("av", "rope_pos", "video_only")
TENSORS, META = load_fixture(TINY / "dit", backend="mlx")


def tiny_model() -> Kandinsky6Transformer3DModel:
    model = Kandinsky6Transformer3DModel.from_config(META["config"])
    load_module_weights(model, TENSORS, "weights")
    return model


def model_inputs(scenario: str) -> dict:
    inputs = {k: v for k, v in select(TENSORS, f"inputs.{scenario}").items() if "@" not in k}
    if "visual_rope_pos.0" in inputs:
        inputs["visual_rope_pos"] = tuple(inputs.pop(f"visual_rope_pos.{i}") for i in range(3))
    return inputs


def reference_outputs(scenario: str) -> list:
    names = ("sample", "audio_sample")
    return [TENSORS[f"outputs.{scenario}.{name}"] for name in names if f"outputs.{scenario}.{name}" in TENSORS]


class _Recording:
    """Stands in for a block in the model's block list and keeps the output of every call."""

    def __init__(self, block, name: str, sink: dict):
        self.block, self.name, self.sink = block, name, sink

    def __call__(self, *args):
        output = self.block(*args)
        self.sink[self.name] = output
        return output


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_final_outputs_match_reference(scenario):
    outputs = tiny_model()(**model_inputs(scenario))
    references = reference_outputs(scenario)
    assert len(outputs) == len(references)  # (video,) without audio, (video, audio) with it
    for got, ref in zip(outputs, references):
        assert compare(got, ref)["rel_max"] <= 1e-4


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_block_output_inside_the_model_matches_reference(scenario):
    model = tiny_model()
    sink: dict = {}
    for attr in ("video_text_transformer_blocks", "audio_text_transformer_blocks", "visual_transformer_blocks"):
        blocks = getattr(model, attr)
        for i, block in enumerate(blocks):
            blocks[i] = _Recording(block, f"{attr}.{i}", sink)
    model(**model_inputs(scenario))

    calls = [c for c in META["calls"][scenario] if c["class"].endswith(("EncoderBlock", "DecoderBlock"))]
    assert len(calls) == len(sink) == 2 * META["config"]["num_text_blocks"] + META["config"]["num_visual_blocks"]
    for call in calls:
        got, ref = sink[call["module"]], unflatten(call["output"], TENSORS)
        pairs = zip(got, ref) if isinstance(ref, tuple) else [(got, ref)]
        for g, r in pairs:
            assert (g is None) == (r is None), call["module"]
            if r is not None:
                assert compare(g, r)["rel_max"] <= 1e-4, call["module"]


def test_token_type_ids_need_embeddings():
    config = META["config"] | {"visual_token_type_num_embeddings": 0}
    model = Kandinsky6Transformer3DModel.from_config(config)
    with pytest.raises(ValueError, match="visual_token_type_num_embeddings"):
        model(**model_inputs("av"))


def _write_converted(directory: Path, bf16_modulation: bool) -> None:
    """A converted transformer directory like the converter writes, with modulation stored as BF16 like Lite."""
    weights = {}
    for name, value in select(TENSORS, "weights").items():
        weights[name] = value.astype(mx.bfloat16) if bf16_modulation and "modulation" in name else value
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    (directory / "config.json").write_text(json.dumps(META["config"] | {"is_multimodal": True}))


def test_from_converted_float32_matches_reference(tmp_path):
    _write_converted(tmp_path, bf16_modulation=False)
    model = Kandinsky6Transformer3DModel.from_converted(tmp_path, dtype=mx.float32)
    for got, ref in zip(model(**model_inputs("av")), reference_outputs("av")):
        assert compare(got, ref)["rel_max"] <= 1e-4


def test_from_converted_bf16_keeps_time_embeddings_and_modulation_in_float32(tmp_path):
    _write_converted(tmp_path, bf16_modulation=True)
    model = Kandinsky6Transformer3DModel.from_converted(tmp_path, dtype=mx.bfloat16)
    shapes = parameter_shapes(model)
    dtypes = {name: value.dtype for name, value in tree_flatten(model.parameters())}
    assert set(dtypes) == set(shapes)
    kept = [name for name in dtypes if keeps_fp32("transformer", name)]
    assert any("time_embeddings" in name for name in kept) and any("modulation" in name for name in kept)
    for name, dtype in dtypes.items():
        assert dtype == (mx.float32 if keeps_fp32("transformer", name) else mx.bfloat16), name

    outputs = model(**model_inputs("av"))
    assert all(out.dtype == mx.bfloat16 for out in outputs)
    for got, ref in zip(outputs, reference_outputs("av")):
        assert compare(got, ref)["cosine"] >= 0.999


@pytest.mark.parametrize("name", ["lite-distill", "pro-distill"])
def test_real_config_builds_reference_parameter_names_and_shapes(name):
    torch = pytest.importorskip("torch")
    from diffusers import Kandinsky6Transformer3DModel as Reference

    config = json.loads((CONFIGS / f"{name}-transformer.json").read_text())
    with torch.device("meta"):
        reference = Reference.from_config(config)
    expected = {key: tuple(value.shape) for key, value in reference.state_dict().items()}
    # Built lazily: MLX does not allocate the random initial parameters unless they are evaluated.
    got = parameter_shapes(Kandinsky6Transformer3DModel.from_config(config))
    assert got == expected
    assert sum(int(np.prod(shape)) for shape in got.values()) == sum(p.numel() for p in reference.parameters())
