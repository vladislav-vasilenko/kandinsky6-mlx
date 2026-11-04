"""Storage dtype of converted tensors, and which DiT parameters run in float32.

Storage modes:
- `"checkpoint"` (default): every tensor keeps its stored dtype, so BF16 data stays byte-identical and the file does
  not grow. In the Lite DiT, 268 modulation tensors (549M parameters) are stored as BF16; diffusers upcasts them when
  loading, and storing them as float32 would add 1.02 GiB without adding information.
- `"float32"`: every floating tensor is stored as float32, for parity tests against a float32 reference.

Compute: diffusers keeps `_keep_in_fp32_modules = ["time_embeddings", "modulation"]` of the DiT in float32. The MLX
model applies `keeps_fp32` when it loads weights (spec 05); the converter does not need to.
"""

from dataclasses import dataclass

KEEP_FP32_SUBSTRINGS = {"transformer": ("time_embeddings", "modulation")}
FLOAT_DTYPES = ("F32", "BF16", "F16", "F64")
MODES = ("checkpoint", "float32")


def keeps_fp32(component: str, name: str, rule: str = "substring") -> bool:
    """Whether parameter `name` of `component` runs in float32 regardless of the requested dtype.

    `rule="substring"` holds every time embedding and modulation in float32, the port's default. `rule="diffusers"` is what
    `from_pretrained(torch_dtype=...)` does in diffusers 0.42.1, which compares each dot-separated part of the name exactly
    (`models/model_loading_utils.py`), so only `out_layer.modulation` and `audio_out_layer.modulation` match.
    """
    parts = KEEP_FP32_SUBSTRINGS.get(component, ())
    if rule == "diffusers":
        return any(part in name.split(".") for part in parts)
    if rule == "substring":
        return any(part in name for part in parts)
    raise ValueError(f"unknown float32 rule {rule!r}; expected 'substring' or 'diffusers'")


@dataclass(frozen=True)
class DtypePolicy:
    mode: str = "checkpoint"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown dtype policy {self.mode!r}; expected one of {MODES}")

    def target(self, component: str, name: str, stored: str) -> str:
        """Safetensors dtype tag (`"F32"`, `"BF16"`, ...) to store tensor `name` in."""
        if self.mode == "float32" and stored in FLOAT_DTYPES:
            return "F32"
        return stored
