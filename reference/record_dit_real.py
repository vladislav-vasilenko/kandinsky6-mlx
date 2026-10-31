"""Record the real Lite-distill DiT on Smoke-size inputs for spec 05 stage B.

Produces one fixture (`--out`, default `fixtures/real/dit_smoke`) with:

- `inputs.ref.*`: the model inputs. Video latents `[1, 16, 7, 32, 54]` (`B, C, T, H, W`, 25 frames at 256x432) from
  `np.random.default_rng(seed)`, permuted to `(B, T, H, W, C)` and widened to 33 channels with the zero conditioning
  latent and zero mask the T2V pipeline appends; audio latents `[1, 45, 40]`; the real text embeddings; and a
  mid-trajectory timestep (`1000 * sigma` at `s = 0.5` with the released shift 5, i.e. 833.33).
- the float32 output of every encoder and fused block, and the final outputs (`outputs.ref.out.*`). Only the first
  block of each chain keeps its full recorded inputs; the call of every later block points its stream inputs at the
  previous block's output, so the file stays near 0.8 GB instead of 2 GB. `args_to_model_dtype` in each call lists
  the arguments a bf16 run casts to bf16 (time embeddings and RoPE tables stay float32, as in diffusers).
- `meta["torch_bf16"]`: the tolerance basis. `per_block` replays every block in bf16 (diffusers `from_pretrained`
  with `torch_dtype=torch.bfloat16`, which keeps time embeddings and modulation in float32) on the float32 inputs and
  gives `rel_l2`/`cosine` against the float32 outputs; `chained` runs the whole bf16 model and compares every block
  output; `final` compares the final outputs. `device` records where the bf16 runs ran.

Text embeddings come from `--text-embeds` (a fixture from `reference/encode_text.py`) or from `--prompt`, encoded here
on the CPU (slow; prefer encode_text.py on a GPU) and saved to `--save-text-embeds`.

The float32 reference is recorded on the CPU (about 12.5 GB of RAM). bf16 on a CPU without native bf16 is extremely
slow, so the drift can be split off: record with `--skip-bf16`, then add the drift to the same fixture later with
`--drift-only --device mps` (Apple GPU) or `--device cuda`; it then compares a GPU bf16 run with the CPU float32
reference, which is recorded in the report.

Running on Kaggle (an outward action: prepare it, then run only with the owner's approval): `kaggle/encode_text/`
encodes the Smoke prompt on a GPU, and `kaggle/record_dit/` records the float32 reference on the CPU from those
embeddings with `--skip-bf16`. The Smoke prompt is the study notebook's `orbit_woman` preset, `smoke` profile
(`applehawk/kandinsky-6-lite-diffusers-generation`, `compile_prompt`), stored in `fixtures/real_inputs/smoke_prompt.txt`.
"""

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.utils.fixtures import save_fixture
from reference.encode_text import encode_prompt, find_component_root
from reference.recorder import ActivationRecorder, flatten, library_versions, modules_of_classes, unflatten

BLOCK_CLASSES = ("Kandinsky6TransformerEncoderBlock", "Kandinsky6FusedTransformerDecoderBlock")
# Arguments of each block class a bf16 run casts to bf16: the streams and text states. Time embeddings (fused args[4],
# encoder args[1]) and RoPE tables stay float32 in diffusers, and the mask is boolean.
MODEL_DTYPE_ARGS = {"Kandinsky6TransformerEncoderBlock": [0], "Kandinsky6FusedTransformerDecoderBlock": [0, 1, 2, 3]}
LATENT_SHAPE = (1, 16, 7, 32, 54)  # B, C, T, H, W for 256x432, 25 frames
AUDIO_SHAPE = (1, 45, 40)
SHIFT = 5.0


def mid_trajectory_timestep(s: float = 0.5, shift: float = SHIFT) -> float:
    return 1000.0 * shift * s / (1.0 + (shift - 1.0) * s)


def load_text_embeds(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    tensors = load_file(str(path.with_suffix(".safetensors")))
    names = ("encoder_hidden_states", "pooled_projections", "encoder_attention_mask")
    return {name: tensors[f"inputs.{name}"] for name in names if f"inputs.{name}" in tensors}


def smoke_inputs(seed: int, text: dict[str, torch.Tensor]) -> dict:
    rng = np.random.default_rng(seed)
    latents = torch.from_numpy(rng.standard_normal(LATENT_SHAPE).astype(np.float32)).permute(0, 2, 3, 4, 1)
    cond_mask = torch.zeros((*latents.shape[:-1], 1))
    audio = torch.from_numpy(rng.standard_normal(AUDIO_SHAPE).astype(np.float32))
    return {
        "hidden_states": torch.cat([latents, torch.zeros_like(latents), cond_mask], dim=-1).contiguous(),
        "audio_hidden_states": audio,
        "timestep": torch.tensor([mid_trajectory_timestep()], dtype=torch.float32),
        **text,
    }


def chain_calls(calls: list[dict], tensors: dict) -> list[dict]:
    """Point the stream inputs of every block after the first in its chain at the previous block's output, drop
    their duplicated recorded inputs, and note which arguments a bf16 run casts."""
    chained, previous = [], {}
    for call in calls:
        call = copy.deepcopy(call)
        chain = call["module"].rsplit(".", 1)[0]
        args = call["args"]["tuple"]
        if chain in previous:
            first, prev = previous[chain]
            for key in [k for k in tensors if k.startswith(f"inputs.ref.{call['module']}@")]:
                del tensors[key]
            outputs = prev["output"]["tuple"] if "tuple" in prev["output"] else [prev["output"]]
            args = outputs + copy.deepcopy(first["args"]["tuple"][len(outputs) :])
            call["args"] = {"tuple": args}
        else:
            first = call
        previous[chain] = (first, call)
        call["args_to_model_dtype"] = MODEL_DTYPE_ARGS[call["class"]]
        chained.append(call)
    return chained


def block_inputs(call: dict, tensors: dict, dtype: torch.dtype) -> tuple:
    args = list(unflatten(call["args"], tensors))
    for index in call["args_to_model_dtype"]:
        if args[index] is not None:
            args[index] = args[index].to(dtype)
    return tuple(args)


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a, r = actual.double().flatten(), reference.double().flatten()
    return {
        "rel_l2": float((a - r).norm() / r.norm()),
        "cosine": float(torch.dot(a, r) / (a.norm() * r.norm())),
        "rel_max": float((a - r).abs().max() / r.abs().max()),
    }


def output_metrics(actual, reference) -> dict:
    """Metrics of a block output; fused blocks give one entry per present modality."""
    if isinstance(reference, tuple):
        names = ("video", "audio")
        return {n: metrics(a, r) for n, a, r in zip(names, actual, reference) if r is not None}
    return {"text": metrics(actual, reference)}


def load_transformer(checkpoint: Path, dtype: torch.dtype, device: str = "cpu"):
    from diffusers import Kandinsky6Transformer3DModel

    path = find_component_root(checkpoint, "transformer") / "transformer"
    return Kandinsky6Transformer3DModel.from_pretrained(path, torch_dtype=dtype).eval().to(device)


def record_reference(checkpoint: Path, kwargs: dict) -> tuple[dict, list]:
    """Float32 CPU reference: every block output and the final outputs, with chained block inputs."""
    model = load_transformer(checkpoint, torch.float32)
    modules = modules_of_classes(model, BLOCK_CLASSES)
    tensors: dict = {}
    for key, value in kwargs.items():
        flatten(value, f"inputs.ref.{key}", tensors)
    with ActivationRecorder(model, modules, tag="ref") as recorder, torch.no_grad():
        output = model(**kwargs, return_dict=False)
    tensors.update(recorder.tensors)
    flatten(tuple(output), "outputs.ref.out", tensors)
    calls = chain_calls(recorder.calls, tensors)
    del model, recorder
    gc.collect()
    return tensors, calls


def bf16_drift(checkpoint: Path, tensors: dict, calls: list, device: str = "cpu") -> dict:
    """The tolerance basis: torch bf16 (time embeddings and modulation float32, as diffusers loads it) vs the float32
    recording, per block on the recorded float32 inputs, per block inside the chained bf16 forward, and final."""
    model = load_transformer(checkpoint, torch.bfloat16, device)
    named = dict(model.named_modules())
    modules = [call["module"] for call in calls]

    def on_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (tuple, list)):  # the fused block's (t_v, t_a)
            return type(obj)(on_device(o) for o in obj)
        return obj

    per_block = {}
    with torch.no_grad():
        for call in calls:
            args = tuple(on_device(a) for a in block_inputs(call, tensors, torch.bfloat16))
            got = named[call["module"]](*args)
            got = tuple(None if g is None else g.cpu() for g in got) if isinstance(got, tuple) else got.cpu()
            per_block[call["module"]] = output_metrics(got, unflatten(call["output"], tensors))
        kwargs = {
            k[len("inputs.ref.") :]: v for k, v in tensors.items() if k.startswith("inputs.ref.") and "@" not in k
        }
        bf16_kwargs = {
            k: (v.to(torch.bfloat16) if v.is_floating_point() and k != "timestep" else v).to(device)
            for k, v in kwargs.items()
        }
        with ActivationRecorder(model, modules, tag="bf16") as recorder:
            bf16_output = model(**bf16_kwargs, return_dict=False)
    chained = {
        call["module"]: output_metrics(
            unflatten(bf16_call["output"], recorder.tensors), unflatten(call["output"], tensors)
        )
        for call, bf16_call in zip(calls, recorder.calls)
    }
    reference = [tensors[f"outputs.ref.out.{i}"] for i in range(len(bf16_output))]
    final = {n: metrics(a.cpu(), r) for n, a, r in zip(("video", "audio"), bf16_output, reference)}
    return {"per_block": per_block, "chained": chained, "final": final, "device": device}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="diffusers snapshot (with transformer/)")
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--text-embeds", type=Path, help="fixture with the cached text embeddings")
    text.add_argument("--prompt", help="encode this prompt with the checkpoint's text encoders")
    parser.add_argument(
        "--text-checkpoint", type=Path, help="root with text_encoder*/tokenizer* (default: --checkpoint)"
    )
    parser.add_argument("--save-text-embeds", type=Path, help="where to save embeddings computed from --prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bf16", action="store_true", help="record the float32 reference only")
    parser.add_argument(
        "--drift-only", action="store_true", help="add the bf16 drift to an existing recording at --out"
    )
    parser.add_argument("--device", default="cpu", help="device of the bf16 drift runs (cpu, cuda, mps)")
    parser.add_argument("--out", type=Path, default=Path("fixtures/real/dit_smoke"))
    args = parser.parse_args(argv)

    if args.drift_only:
        from safetensors.torch import load_file

        tensors = load_file(str(args.out.with_suffix(".safetensors")))
        meta_path = args.out.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta["torch_bf16"] = bf16_drift(args.checkpoint, tensors, meta["calls"]["ref"], args.device)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        print(f"added the bf16 drift ({args.device}) to {meta_path}")
        return 0

    if args.prompt is not None:
        text_inputs = encode_prompt(args.text_checkpoint or args.checkpoint, args.prompt)
        if args.save_text_embeds:
            save_fixture(
                args.save_text_embeds,
                {f"inputs.{k}": v for k, v in text_inputs.items()},
                {"fixture": args.save_text_embeds.name, "prompt": args.prompt, "versions": library_versions()},
            )
    elif args.text_embeds is not None:
        text_inputs = load_text_embeds(args.text_embeds)
    else:
        parser.error("one of --text-embeds or --prompt is required")
    kwargs = smoke_inputs(args.seed, text_inputs)
    tensors, calls = record_reference(args.checkpoint, kwargs)
    meta = {
        "fixture": args.out.name,
        "checkpoint": str(args.checkpoint),
        "reference": "diffusers.Kandinsky6Transformer3DModel.forward",
        "dtype": "float32",
        "seed": args.seed,
        "inputs": {
            "latents_bcthw": list(LATENT_SHAPE),
            "audio": list(AUDIO_SHAPE),
            "timestep": kwargs["timestep"].item(),
        },
        "versions": library_versions() | {"torch": torch.__version__},
        "calls": {"ref": calls},
    }
    if not args.skip_bf16:
        meta["torch_bf16"] = bf16_drift(args.checkpoint, tensors, calls, args.device)
    save_fixture(args.out, tensors, meta)
    print(f"recorded {len(calls)} block calls, {len(tensors)} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
