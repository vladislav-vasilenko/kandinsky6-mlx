"""Replay recorded video VAE decoder calls through the MLX port.

Fixture weights are in the PyTorch layout and are moved to the MLX layout with the converter's own rules. Recorded
tensors are `(B, C, T, H, W)`; MLX submodules take `(B, T, H, W, C)`, so inputs and outputs are transposed at every
call. The mid-block `Attention` is recorded on flattened tokens with an explicit mask; it is covered through the mid
block and by a separate mask test instead of being replayed alone.
"""

from kandinsky6_mlx.models.video_vae.decoder import AutoencoderKLHunyuanVideo
from kandinsky6_mlx.utils.compare import compare
from kandinsky6_mlx.utils.fixtures import load_fixture, select, unflatten
from tests.parity.audio_cases import submodule, to_mlx_layout
from tests.parity.dit_cases import TINY

SKIPPED_CLASSES = ("Attention",)


def tiny_video_vae() -> tuple[AutoencoderKLHunyuanVideo, dict, dict]:
    tensors, meta = load_fixture(TINY / "video_vae_decoder", backend="mlx")
    model = AutoencoderKLHunyuanVideo.from_config(meta["config"])
    weights = to_mlx_layout("vae", select(tensors, "weights"))
    model.load_weights(AutoencoderKLHunyuanVideo.decoder_weights(weights), strict=True)
    return model, tensors, meta


def video_vae_results() -> list[dict]:
    model, tensors, meta = tiny_video_vae()
    results = []
    for call in meta["calls"]["decode"]:
        if call["class"] in SKIPPED_CLASSES:
            continue
        (x,) = unflatten(call["args"], tensors)
        output = submodule(model, call["module"])(x.transpose(0, 2, 3, 4, 1)).transpose(0, 4, 1, 2, 3)
        metrics = compare(output, unflatten(call["output"], tensors))
        results.append({"module": call["module"], "class": call["class"], "rel_max": metrics["rel_max"]})
    return results
