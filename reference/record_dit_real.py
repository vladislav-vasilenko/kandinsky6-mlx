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
  output and the final outputs; the final bf16 outputs are stored as `outputs.bf16.out.*`.

Text embeddings come from `--text-embeds` (a fixture with `inputs.encoder_hidden_states`, `inputs.pooled_projections`
and optionally `inputs.encoder_attention_mask`) or are computed from `--prompt` with the checkpoint's Qwen2.5-VL and
CLIP encoders (in bf16, upcast to float32; their precision does not matter for DiT parity because both sides get the
same arrays) and saved to `--save-text-embeds` for reuse by spec 09.

Memory: the float32 model is ~12.5 GB and is freed before the bf16 model (~6.3 GB) is loaded; with `--prompt`, Qwen in
bf16 (~15.5 GB) is loaded and freed first. A Kaggle CPU session (30 GB RAM) is enough.

Running on Kaggle (an outward action: prepare it, then run only with the owner's approval):

1. `git archive --format=zip -o k6-mlx-src.zip HEAD` and upload it as a private Kaggle dataset; it gives this script,
   `reference/recorder.py` and `src/kandinsky6_mlx/utils/fixtures.py` (none of them needs MLX).
2. Create a CPU notebook with that dataset and the outputs of `applehawk/k6-lite-diffusers-weights-media` and
   `applehawk/k6-lite-diffusers-weights-text` (Lite-distill revision 4c487e8b) attached.
3. `pip install diffusers==0.42.1 transformers==5.22.1 accelerate==1.13.0`, then

       python <src>/reference/record_dit_real.py --checkpoint /kaggle/input --text-checkpoint /kaggle/input \\
           --prompt "$(cat <src>/fixtures/real_inputs/smoke_prompt.txt)" \\
           --save-text-embeds /kaggle/working/smoke_text --out /kaggle/working/dit_smoke

   The Smoke prompt is the study notebook's `orbit_woman` preset, `smoke` profile
   (`applehawk/kandinsky-6-lite-diffusers-generation`, `compile_prompt`), stored in
   `fixtures/real_inputs/smoke_prompt.txt`.

4. Download `dit_smoke.{safetensors,json}` and `smoke_text.{safetensors,json}` into `fixtures/real/` on the Mac.
"""

import argparse
import copy
import gc
import inspect
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.utils.fixtures import save_fixture
from reference.recorder import ActivationRecorder, flatten, library_versions, modules_of_classes, unflatten

BLOCK_CLASSES = ("Kandinsky6TransformerEncoderBlock", "Kandinsky6FusedTransformerDecoderBlock")
TEXT_COMPONENTS = ("text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2")
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


def find_component_root(root: Path, component: str) -> Path:
    """The directory holding `component/config.json` under `root` (Kaggle mounts outputs at varying depths)."""
    if (root / component / "config.json").is_file():
        return root
    for config in sorted(root.rglob(f"{component}/config.json")):
        return config.parent.parent
    raise FileNotFoundError(f"no {component}/config.json under {root}")


def encode_prompt(text_checkpoint: Path, prompt: str) -> dict[str, torch.Tensor]:
    """`Kandinsky6TI2VAPipeline.encode_prompt` with only the text components built, like the study notebook."""
    from diffusers import Kandinsky6TI2VAPipeline
    from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    root = find_component_root(text_checkpoint, "text_encoder")
    others = inspect.signature(Kandinsky6TI2VAPipeline.__init__).parameters
    pipe = Kandinsky6TI2VAPipeline(
        **{name: None for name in others if name not in ("self", *TEXT_COMPONENTS)},
        text_encoder=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            root / "text_encoder", dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).eval(),
        tokenizer=Qwen2_5_VLProcessor.from_pretrained(root / "tokenizer"),
        text_encoder_2=CLIPTextModel.from_pretrained(root / "text_encoder_2", dtype=torch.float32).eval(),
        tokenizer_2=CLIPTokenizer.from_pretrained(root / "tokenizer_2"),
    )
    with torch.no_grad():
        embeds, pooled, mask, *_ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, dtype=torch.float32)
    del pipe
    gc.collect()
    text = {"encoder_hidden_states": embeds.float(), "pooled_projections": pooled.float()}
    if mask is not None:
        text["encoder_attention_mask"] = mask
    return text


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


def load_transformer(checkpoint: Path, dtype: torch.dtype):
    from diffusers import Kandinsky6Transformer3DModel

    path = find_component_root(checkpoint, "transformer") / "transformer"
    return Kandinsky6Transformer3DModel.from_pretrained(path, torch_dtype=dtype).eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="diffusers snapshot (with transformer/)")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text-embeds", type=Path, help="fixture with the cached text embeddings")
    text.add_argument("--prompt", help="encode this prompt with the checkpoint's text encoders")
    parser.add_argument(
        "--text-checkpoint", type=Path, help="root with text_encoder*/tokenizer* (default: --checkpoint)"
    )
    parser.add_argument("--save-text-embeds", type=Path, help="where to save embeddings computed from --prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("fixtures/real/dit_smoke"))
    args = parser.parse_args(argv)

    if args.prompt is not None:
        text_inputs = encode_prompt(args.text_checkpoint or args.checkpoint, args.prompt)
        if args.save_text_embeds:
            save_fixture(
                args.save_text_embeds,
                {f"inputs.{k}": v for k, v in text_inputs.items()},
                {"fixture": args.save_text_embeds.name, "prompt": args.prompt, "versions": library_versions()},
            )
    else:
        text_inputs = load_text_embeds(args.text_embeds)
    kwargs = smoke_inputs(args.seed, text_inputs)

    # 1. Float32 reference: every block output and the final outputs.
    model = load_transformer(args.checkpoint, torch.float32)
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

    # 2. Torch bf16 drift, the tolerance basis: each block on the float32 inputs, then the chained bf16 model.
    model = load_transformer(args.checkpoint, torch.bfloat16)
    named = dict(model.named_modules())
    per_block = {}
    with torch.no_grad():
        for call in calls:
            got = named[call["module"]](*block_inputs(call, tensors, torch.bfloat16))
            per_block[call["module"]] = output_metrics(got, unflatten(call["output"], tensors))
        bf16_kwargs = {
            k: v.to(torch.bfloat16) if v.is_floating_point() and k != "timestep" else v
            for k, v in kwargs.items()
            if isinstance(v, torch.Tensor)
        }
        with ActivationRecorder(model, modules, tag="bf16") as recorder:
            bf16_output = model(**bf16_kwargs, return_dict=False)
    chained = {
        call["module"]: output_metrics(
            unflatten(bf16_call["output"], recorder.tensors), unflatten(call["output"], tensors)
        )
        for call, bf16_call in zip(calls, recorder.calls)
    }
    names = ("video", "audio")
    final = {n: metrics(a, r) for n, a, r in zip(names, bf16_output, output)}
    flatten(tuple(o.float() for o in bf16_output), "outputs.bf16.out", tensors)

    save_fixture(
        args.out,
        tensors,
        {
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
            "versions": library_versions(),
            "calls": {"ref": calls},
            "torch_bf16": {"per_block": per_block, "chained": chained, "final": final},
        },
    )
    print(f"recorded {len(calls)} block calls, {len(tensors)} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
