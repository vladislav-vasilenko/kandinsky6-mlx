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

TOLERANCE = {"Kandinsky6RoPE1D": 1e-6, "Kandinsky6RoPE3D": 1e-6, "Kandinsky6Attention": 1e-4, "FeedForward": 1e-4}


def replay_calls(
    fixture: str, scenario: str, classes: tuple[str, ...], build: Callable[[str, str, dict, dict], nn.Module]
) -> list[dict]:
    tensors, meta = load_fixture(TINY / fixture, backend="mlx")
    results = []
    for call in meta["calls"][scenario]:
        if call["class"] not in classes:
            continue
        module = build(call["class"], call["module"], tensors, meta["config"])
        load_module_weights(module, tensors, f"weights.{call['module']}")
        args = unflatten(call["args"], tensors)
        kwargs = unflatten(call["kwargs"], tensors)
        output = module(*args, **kwargs)
        reference = unflatten(call["output"], tensors)
        metrics = compare(output, reference)
        results.append(
            {
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


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    results = primitive_results()
    worst: dict[str, float] = {}
    for result in results:
        worst[result["class"]] = max(worst.get(result["class"], 0.0), result["rel_max"])
    print(json.dumps({"calls": len(results), "worst_rel_max_per_class": worst}, indent=2))
