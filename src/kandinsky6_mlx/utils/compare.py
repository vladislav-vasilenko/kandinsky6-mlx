"""Metrics for comparing an MLX result with a reference result."""

from typing import Any

import numpy as np


def to_numpy(array: Any) -> np.ndarray:
    """Convert an MLX array, torch tensor or array-like to a float64 NumPy array."""
    module = type(array).__module__
    if module.startswith("mlx"):
        import mlx.core as mx

        return np.array(array.astype(mx.float32)).astype(np.float64)
    if module.startswith("torch"):
        return array.detach().to("cpu").float().numpy().astype(np.float64)
    return np.asarray(array, dtype=np.float64)


def _dtype_name(array: Any) -> str:
    dtype = array.dtype if hasattr(array, "dtype") else np.asarray(array).dtype
    return str(dtype).rsplit(".", 1)[-1]


def compare(actual: Any, reference: Any) -> dict:
    """Compare `actual` with `reference` in float64.

    Returns `max_abs`, `rel_max = max_abs / max(|reference|)`, `rel_l2 = ||actual - reference|| / ||reference||`,
    `cosine` of the flattened arrays, plus shapes and dtypes. Shapes must be equal.
    """
    a, r = to_numpy(actual), to_numpy(reference)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch: actual {a.shape}, reference {r.shape}")
    diff = a - r
    ref_max = float(np.abs(r).max()) if r.size else 0.0
    ref_norm = float(np.linalg.norm(r))
    a_norm = float(np.linalg.norm(a))
    max_abs = float(np.abs(diff).max()) if diff.size else 0.0
    if a_norm == 0.0 and ref_norm == 0.0:
        cosine = 1.0
    elif a_norm == 0.0 or ref_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(a.ravel(), r.ravel()) / (a_norm * ref_norm))
    return {
        "max_abs": max_abs,
        "rel_max": max_abs / ref_max if ref_max else max_abs,
        "rel_l2": float(np.linalg.norm(diff)) / ref_norm if ref_norm else float(np.linalg.norm(diff)),
        "cosine": cosine,
        "shape": list(a.shape),
        "dtype": {"actual": _dtype_name(actual), "reference": _dtype_name(reference)},
    }
