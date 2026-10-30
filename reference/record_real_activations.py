"""Run a reference component on given inputs and record its output and the activations of chosen submodules.

Meant for real checkpoints (typically on Kaggle, then the result is copied to `fixtures/real/`). It works the same
on a tiny model saved with `save_pretrained`, which is how it is tested.

Usage:
    uv run --group reference python reference/record_real_activations.py \
        --checkpoint <snapshot dir> --component transformer --inputs inputs.safetensors \
        --modules blocks --dtype bfloat16 --out fixtures/real/dit_smoke

`--inputs` is a fixture-style safetensors file whose `inputs.<name>` tensors become keyword arguments of the entry
point (`forward` for the transformer and vocoder, `decode` for the VAEs). Keys `inputs.<name>.0`, `inputs.<name>.1`,
... are passed as a tuple (e.g. `visual_rope_pos`). `--modules` takes module names, or `blocks` for every
transformer/VAE/vocoder block, or `none`.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.utils.fixtures import load_fixture, save_fixture
from reference.recorder import ActivationRecorder, flatten, library_versions, modules_of_classes

COMPONENTS = {
    "transformer": ("Kandinsky6Transformer3DModel", "forward"),
    "vae": ("AutoencoderKLHunyuanVideo", "decode"),
    "audio_vae": ("MMAudioVAE", "decode"),
    "vocoder": ("MMAudioVocoder", "forward"),
}
BLOCK_CLASSES = {
    "transformer": ["Kandinsky6TransformerEncoderBlock", "Kandinsky6FusedTransformerDecoderBlock"],
    "vae": ["HunyuanVideoMidBlock3D", "HunyuanVideoUpBlock3D"],
    "audio_vae": ["MMAudioResnetBlock1D", "MMAudioAttnBlock1D"],
    "vocoder": ["MMAudioAMPBlock"],
}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_component(checkpoint: Path, component: str, dtype: torch.dtype) -> torch.nn.Module:
    import diffusers
    from diffusers.models.autoencoders.autoencoder_mmaudio import MMAudioVAE
    from diffusers.models.autoencoders.mmaudio_vocoder import MMAudioVocoder

    class_name, _ = COMPONENTS[component]
    cls = {"MMAudioVAE": MMAudioVAE, "MMAudioVocoder": MMAudioVocoder}.get(class_name) or getattr(diffusers, class_name)
    path = checkpoint / component if (checkpoint / component).is_dir() else checkpoint
    return cls.from_pretrained(path, torch_dtype=dtype).eval()


def read_inputs(path: Path, dtype: torch.dtype) -> dict:
    tensors, _ = load_fixture(path)
    grouped: dict[str, dict] = {}
    for key, value in tensors.items():
        if not key.startswith("inputs."):
            continue
        name, _, index = key[len("inputs.") :].partition(".")
        tensor = torch.from_numpy(value.copy())
        if tensor.is_floating_point() and name not in ("timestep",):
            tensor = tensor.to(dtype)
        grouped.setdefault(name, {})[index] = tensor
    kwargs = {}
    for name, parts in grouped.items():
        kwargs[name] = parts[""] if "" in parts else tuple(parts[i] for i in sorted(parts, key=int))
    return kwargs


def record(model: torch.nn.Module, component: str, kwargs: dict, modules: list[str], tag: str) -> tuple[dict, list]:
    _, method = COMPONENTS[component]
    tensors: dict = {}
    for key, value in kwargs.items():
        flatten(value, f"inputs.{tag}.{key}", tensors)
    with ActivationRecorder(model, modules, tag=tag) as recorder, torch.no_grad():
        output = getattr(model, method)(**kwargs, return_dict=False)
    tensors.update(recorder.tensors)
    flatten(tuple(output), f"outputs.{tag}.out", tensors)
    return tensors, recorder.calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--component", choices=sorted(COMPONENTS), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--modules", nargs="+", default=["blocks"])
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    dtype = DTYPES[args.dtype]
    model = load_component(args.checkpoint, args.component, dtype)
    if args.modules == ["blocks"]:
        modules = modules_of_classes(model, BLOCK_CLASSES[args.component])
    elif args.modules == ["none"]:
        modules = []
    else:
        modules = args.modules
    kwargs = read_inputs(args.inputs, dtype)
    tensors, calls = record(model, args.component, kwargs, modules, args.tag)
    save_fixture(
        args.out,
        tensors,
        {
            "fixture": args.out.name,
            "component": args.component,
            "checkpoint": str(args.checkpoint),
            "dtype": args.dtype,
            "reference": f"diffusers.{COMPONENTS[args.component][0]}.{COMPONENTS[args.component][1]}",
            "versions": library_versions(),
            "calls": {args.tag: calls},
        },
    )
    print(f"recorded {len(calls)} calls, {len(tensors)} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
