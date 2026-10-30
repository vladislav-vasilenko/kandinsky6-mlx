"""Fixture files: one safetensors file with tensors plus a JSON sidecar with metadata.

Tensor keys use the prefixes `inputs.`, `outputs.` and `weights.`. Tensors keep the dtype the reference computed them
in; bfloat16 converts to float32 without loss, so `load_fixture(..., backend="numpy")` returns float32 for them.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

PREFIXES = ("inputs", "outputs", "weights")


def _paths(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    stem = path.with_suffix("") if path.suffix in (".safetensors", ".json") else path
    return stem.with_suffix(".safetensors"), stem.with_suffix(".json")


def save_fixture(path: str | Path, tensors: dict[str, Any], meta: dict) -> None:
    """Save tensors (NumPy arrays or torch tensors) and metadata. Writing is deterministic."""
    import torch
    from safetensors.torch import save_file

    for key in tensors:
        if key.split(".", 1)[0] not in PREFIXES:
            raise ValueError(f"tensor key {key!r} must start with one of {PREFIXES}")
    converted = {}
    for key, value in tensors.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(value))
        converted[key] = tensor.detach().to("cpu").contiguous()
    tensor_path, meta_path = _paths(path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(tensor_path))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def load_fixture(path: str | Path, backend: str = "numpy") -> tuple[dict[str, Any], dict]:
    """Load a fixture as `(tensors, meta)`. `backend` is `"numpy"` or `"mlx"`."""
    import mlx.core as mx

    tensor_path, meta_path = _paths(path)
    arrays = mx.load(str(tensor_path))
    meta = json.loads(meta_path.read_text())
    if backend == "mlx":
        return arrays, meta
    if backend != "numpy":
        raise ValueError(f"unknown backend {backend!r}")
    out = {}
    for key, value in arrays.items():
        if value.dtype == mx.bfloat16:
            value = value.astype(mx.float32)
        out[key] = np.array(value)
    return out, meta


def select(tensors: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Return the tensors under `prefix.` with the prefix removed."""
    start = prefix + "."
    return {key[len(start) :]: value for key, value in tensors.items() if key.startswith(start)}


def unflatten(spec: Any, tensors: dict[str, Any]) -> Any:
    """Rebuild an argument structure recorded by `reference.recorder.flatten`, with arrays from `tensors`."""
    if "tensor" in spec:
        return tensors[spec["tensor"]]
    if "none" in spec:
        return None
    if "value" in spec:
        return spec["value"]
    if "dict" in spec:
        return {key: unflatten(value, tensors) for key, value in spec["dict"].items()}
    if "tuple" in spec:
        return tuple(unflatten(value, tensors) for value in spec["tuple"])
    if "list" in spec:
        return [unflatten(value, tensors) for value in spec["list"]]
    raise ValueError(f"bad spec {spec}")


def load_module_weights(module: Any, tensors: dict[str, Any], prefix: str, strict: bool = True) -> None:
    """Load the MLX arrays under `prefix.` (e.g. `weights.video_time_embeddings`) into an MLX module."""
    start = prefix + "."
    weights = [(key[len(start) :], value) for key, value in tensors.items() if key.startswith(start)]
    module.load_weights(weights, strict=strict)
