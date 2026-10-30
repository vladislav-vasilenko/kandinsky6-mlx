"""Replay recorded DiT submodule calls through the MLX port and compare with the recorded reference outputs.

Run as a script to print per-module metrics for the report:

    uv run python -m tests.parity.dit_cases
"""

import json
import math
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from mlx import nn

from kandinsky6_mlx.models.dit import blocks as B
from kandinsky6_mlx.models.dit import primitives as P
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture, load_module_weights, unflatten

TINY = Path(__file__).resolve().parents[2] / "fixtures" / "tiny"


def shape(tensors: dict, name: str) -> tuple[int, ...]:
    return tuple(tensors[f"weights.{name}"].shape)


def build_primitive(cls: str, module: str, tensors: dict, config: dict) -> nn.Module:
    """Construct the MLX counterpart of a recorded reference module from its weight shapes and the model config."""
    if cls == "Kandinsky6Attention":
        out_dim, in_dim = shape(tensors, f"{module}.to_query.weight")
        kv_dim = shape(tensors, f"{module}.to_key.weight")[1]
        head_dim = shape(tensors, f"{module}.query_norm.weight")[0]
        assert in_dim == out_dim
        return P.Kandinsky6Attention(out_dim, head_dim, kv_dim)
    if cls == "FeedForward":
        inner, dim = shape(tensors, f"{module}.net.0.proj.weight")
        return P.FeedForward(dim, inner, bias=f"weights.{module}.net.0.proj.bias" in tensors)
    if cls == "Kandinsky6Modulation":
        out, time_dim = shape(tensors, f"{module}.out_layer.weight")
        return P.Kandinsky6Modulation(time_dim, out, 1)
    if cls == "Kandinsky6TextEmbeddings":
        model_dim, text_dim = shape(tensors, f"{module}.in_layer.weight")
        return P.Kandinsky6TextEmbeddings(text_dim, model_dim)
    if cls == "Kandinsky6VisualEmbeddings":
        model_dim, flat = shape(tensors, f"{module}.in_layer.weight")
        patch = tuple(config["patch_size"])
        return P.Kandinsky6VisualEmbeddings(flat // math.prod(patch), model_dim, patch)
    if cls == "Kandinsky6TimeEmbeddings":
        time_dim, model_dim = shape(tensors, f"{module}.timestep_embedder.linear_1.weight")
        return P.Kandinsky6TimeEmbeddings(model_dim, time_dim)
    if cls == "Embedding":
        num, dims = shape(tensors, f"{module}.weight")
        return nn.Embedding(num, dims)
    if cls == "Kandinsky6RoPE3D":
        return P.Kandinsky6RoPE3D(tuple(config["axes_dims"]))
    if cls == "Kandinsky6RoPE1D":
        if module.startswith("audio_text"):
            return P.Kandinsky6RoPE1D(sum(config["axes_dims_a"]))
        if module == "audio_rope_embeddings":
            return P.Kandinsky6RoPE1D(sum(config["axes_dims_a"]), freqs_scaling=config["audio_freqs_scaling"])
        return P.Kandinsky6RoPE1D(sum(config["axes_dims"]))
    raise KeyError(cls)


PRIMITIVE_CLASSES = (
    "Kandinsky6RoPE1D",
    "Kandinsky6RoPE3D",
    "Kandinsky6Attention",
    "FeedForward",
    "Kandinsky6Modulation",
    "Kandinsky6TextEmbeddings",
    "Kandinsky6VisualEmbeddings",
    "Kandinsky6TimeEmbeddings",
    "Embedding",
)

TOLERANCE = {
    "Kandinsky6RoPE1D": 1e-6,
    "Kandinsky6RoPE3D": 1e-6,
    "Kandinsky6Attention": 1e-4,
    "FeedForward": 1e-4,
    "Kandinsky6TransformerEncoderBlock": 1e-4,
    "Kandinsky6FusedTransformerDecoderBlock": 1e-4,
    "Kandinsky6OutLayer": 1e-4,
    "Kandinsky6OutLayerAudio": 1e-4,
}


def _compare_outputs(output, reference) -> dict:
    """Worst `rel_max` and `cosine` over an output that may be a tuple with `None` entries (absent modalities)."""
    if not isinstance(reference, tuple):
        return compare(output, reference)
    assert isinstance(output, tuple) and len(output) == len(reference)
    rel_max, cosine = 0.0, 1.0
    for got, ref in zip(output, reference):
        assert (got is None) == (ref is None), "absent modality differs from the reference"
        if ref is not None:
            metrics = compare(got, ref)
            rel_max, cosine = max(rel_max, metrics["rel_max"]), min(cosine, metrics["cosine"])
    return {"rel_max": rel_max, "cosine": cosine}


def replay_calls(
    fixture: str,
    scenario: str,
    classes: tuple[str, ...],
    build: Callable[[str, str, dict, dict], nn.Module],
    weights_scope: str = "",
    config: dict | None = None,
) -> list[dict]:
    """Rebuild each recorded module of `classes` in MLX, load its weights from `weights.{weights_scope}{module}`, call
    it with the recorded inputs and compare with the recorded output."""
    tensors, meta = load_fixture(TINY / fixture, backend="mlx")
    config = meta["config"] if config is None else config
    results = []
    for call in meta["calls"][scenario]:
        if call["class"] not in classes:
            continue
        name = weights_scope + call["module"]
        module = build(call["class"], name, tensors, config)
        load_module_weights(module, tensors, f"weights.{name}")
        args = unflatten(call["args"], tensors)
        kwargs = unflatten(call["kwargs"], tensors)
        output = module(*args, **kwargs)
        reference = unflatten(call["output"], tensors)
        metrics = _compare_outputs(output, reference)
        results.append(
            {
                "scenario": scenario,
                "module": call["module"],
                "class": call["class"],
                "rel_max": metrics["rel_max"],
                "cosine": metrics["cosine"],
                "tolerance": TOLERANCE.get(call["class"], 1e-5),
            }
        )
    return results


def primitive_results() -> list[dict]:
    return replay_calls("dit", "av", PRIMITIVE_CLASSES, build_primitive)


BLOCK_CLASSES = (
    "Kandinsky6TransformerEncoderBlock",
    "Kandinsky6FusedTransformerDecoderBlock",
    "Kandinsky6OutLayer",
    "Kandinsky6OutLayerAudio",
)


def build_block(cls: str, module: str, tensors: dict, config: dict) -> nn.Module:
    """Construct the MLX counterpart of a recorded block from its weight shapes and the (variant) config flags."""

    def dims(prefix: str, modulation: str) -> tuple[int, int, int, int]:
        model_dim = shape(tensors, f"{prefix}.feed_forward.net.0.proj.weight")[1]
        time_dim = shape(tensors, f"{prefix}.{modulation}.out_layer.weight")[1]
        ff_dim = shape(tensors, f"{prefix}.feed_forward.net.0.proj.weight")[0]
        attention = "attn" if modulation == "text_modulation" else "self_attention"
        head_dim = shape(tensors, f"{prefix}.{attention}.query_norm.weight")[0]
        return model_dim, time_dim, ff_dim, head_dim

    if cls == "Kandinsky6TransformerEncoderBlock":
        return B.Kandinsky6TransformerEncoderBlock(*dims(module, "text_modulation"))
    if cls == "Kandinsky6FusedTransformerDecoderBlock":
        return B.Kandinsky6FusedTransformerDecoderBlock(
            *dims(f"{module}.video_dec_block", "visual_modulation"),
            *dims(f"{module}.audio_dec_block", "visual_modulation"),
            ca_rope=config["ca_rope"],
            cross_gates=config["cross_gates"],
            fix_modulation=config["fix_modulation"],
        )
    out, model_dim = shape(tensors, f"{module}.out_layer.weight")
    time_dim = shape(tensors, f"{module}.modulation.out_layer.weight")[1]
    if cls == "Kandinsky6OutLayer":
        patch = tuple(config["patch_size"])
        return B.Kandinsky6OutLayer(model_dim, time_dim, out // math.prod(patch), patch)
    if cls == "Kandinsky6OutLayerAudio":
        return B.Kandinsky6OutLayerAudio(model_dim, time_dim, out)
    raise KeyError(cls)


def block_results() -> list[dict]:
    """Every recorded block call: the released flags (av, rope_pos, video_only) and the other flag combinations."""
    results = []
    for scenario in ("av", "rope_pos", "video_only"):
        results += replay_calls("dit", scenario, BLOCK_CLASSES, build_block)
    _, meta = load_fixture(TINY / "dit_variants", backend="numpy")
    for tag, flags in meta["variants"].items():
        results += replay_calls(
            "dit_variants", tag, BLOCK_CLASSES, build_block, weights_scope=f"{tag}.", config=meta["config"] | flags
        )
    return results


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    report = {}
    for name, results in (("primitives", primitive_results()), ("blocks", block_results())):
        worst: dict[str, float] = {}
        for result in results:
            key = result["class"] if name == "primitives" else f"{result['class']}[{result['scenario']}]"
            worst[key] = max(worst.get(key, 0.0), result["rel_max"])
        report[name] = {"calls": len(results), "worst_rel_max": worst}
    print(json.dumps(report, indent=2))
