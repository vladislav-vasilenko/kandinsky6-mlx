"""Replay recorded audio VAE decoder and vocoder calls through the MLX port.

The fixtures store weights in the PyTorch layout; they are moved to the MLX layout with the converter's own rules, so
the tests also cover what the converted checkpoint contains. Recorded tensors are `(B, C, L)`; MLX submodules take
`(B, L, C)`, so inputs and outputs are transposed at the boundary of every call.
"""

import mlx.core as mx
from mlx import nn

from kandinsky6_mlx.convert.layouts import conv_rule
from kandinsky6_mlx.models.audio_vae.decoder import MMAudioVAE
from kandinsky6_mlx.models.vocoder.bigvgan import MMAudioVocoder
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture, select, unflatten
from tests.parity.dit_cases import TINY


def to_mlx_layout(component: str, weights: dict[str, mx.array]) -> dict[str, mx.array]:
    out = {}
    for name, value in weights.items():
        rule = conv_rule(component, name)
        out[name] = value.transpose(rule.axes()) if rule is not None else value
    return out


def submodule(model: nn.Module, path: str) -> nn.Module:
    for part in path.split("."):
        model = model[int(part)] if isinstance(model, list) else getattr(model, part)
    return model


def tiny_audio_vae() -> tuple[MMAudioVAE, dict, dict]:
    tensors, meta = load_fixture(TINY / "audio_vae_decoder", backend="mlx")
    model = MMAudioVAE.from_config(meta["config"])
    weights = to_mlx_layout("audio_vae", select(tensors, "weights"))
    model.load_weights(MMAudioVAE.decoder_weights(weights), strict=True)
    return model, tensors, meta


def tiny_vocoder() -> tuple[MMAudioVocoder, dict, dict]:
    tensors, meta = load_fixture(TINY / "vocoder", backend="mlx")
    model = MMAudioVocoder.from_config(meta["config"])
    model.load_weights(list(to_mlx_layout("vocoder", select(tensors, "weights")).items()), strict=True)
    return model, tensors, meta


def _channels_last(x):
    return x.transpose(0, 2, 1) if isinstance(x, mx.array) and x.ndim == 3 else x


def replay(model: nn.Module, tensors: dict, calls: list[dict]) -> list[dict]:
    results = []
    for call in calls:
        module = submodule(model, call["module"])
        args = [_channels_last(a) for a in unflatten(call["args"], tensors)]
        output = module(*args).transpose(0, 2, 1)
        metrics = compare(output, unflatten(call["output"], tensors))
        results.append({"module": call["module"], "class": call["class"], "rel_max": metrics["rel_max"]})
    return results


def audio_vae_results() -> list[dict]:
    model, tensors, meta = tiny_audio_vae()
    return replay(model, tensors, meta["calls"]["decode"])


def vocoder_results() -> list[dict]:
    model, tensors, meta = tiny_vocoder()
    return replay(model, tensors, meta["calls"]["forward"])
