import mlx.core as mx
import numpy as np
import pytest

from kandinsky6_mlx.utils.compare import compare


def test_identical_arrays():
    x = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    result = compare(mx.array(x), x)
    assert result["max_abs"] == 0.0
    assert result["rel_max"] == 0.0
    assert result["rel_l2"] == 0.0
    assert result["cosine"] == pytest.approx(1.0)
    assert result["shape"] == [3, 4]


def test_known_perturbation():
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(10_000)
    noise = rng.standard_normal(10_000)
    noise *= 1e-3 * np.linalg.norm(reference) / np.linalg.norm(noise)
    result = compare(reference + noise, reference)
    assert result["rel_l2"] == pytest.approx(1e-3, rel=1e-9)
    assert result["max_abs"] == pytest.approx(np.abs(noise).max())
    assert result["rel_max"] == pytest.approx(np.abs(noise).max() / np.abs(reference).max())
    assert 0.999 < result["cosine"] < 1.0


def test_bfloat16_and_torch_inputs():
    torch = pytest.importorskip("torch")
    x = np.linspace(-2, 2, 16, dtype=np.float32)
    result = compare(mx.array(x).astype(mx.bfloat16), torch.from_numpy(x))
    assert result["dtype"] == {"actual": "bfloat16", "reference": "float32"}
    assert result["rel_max"] < 1e-2


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare(np.zeros(3), np.zeros(4))
