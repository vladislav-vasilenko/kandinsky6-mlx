import subprocess
import sys

import mlx.core as mx


def test_mlx_available():
    result = mx.array([1.0]) + 1
    mx.eval(result)
    assert result.item() == 2.0


def test_bf16_supported():
    a = mx.ones((4, 4), dtype=mx.bfloat16)
    result = a @ a
    mx.eval(result)
    assert result.dtype == mx.bfloat16
    assert result[0, 0].item() == 4.0


def test_reference_group_optional():
    # A fresh interpreter: other tests may import torch into this process.
    code = "import sys, kandinsky6_mlx; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
