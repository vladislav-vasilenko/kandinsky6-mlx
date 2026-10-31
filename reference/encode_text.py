"""Encode a prompt into Kandinsky 6 text embeddings with the reference Qwen2.5-VL and CLIP encoders.

Runs `Kandinsky6TI2VAPipeline.encode_prompt` (diffusers 0.42.1) with only the text components built, like the study
notebook, and saves a fixture with `inputs.encoder_hidden_states` (padded to `max_sequence_length` tokens),
`inputs.pooled_projections` and `inputs.encoder_attention_mask` (absent when nothing is padded), all float32.

Qwen in bfloat16 on a CPU without native bf16 is extremely slow (over 80 minutes for one prompt on a Kaggle CPU
session without finishing); use a GPU (`--device cuda`, float16 on T4, spread over the visible GPUs) or Apple MPS.
The precision of the embeddings does not matter for parity work, because the MLX side and the reference get the same
arrays.

    python reference/encode_text.py --text-checkpoint <root with text_encoder*/tokenizer*> \\
        --prompt-file fixtures/real_inputs/smoke_prompt.txt --device cuda --dtype float16 --out smoke_text
"""

import argparse
import gc
import inspect
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kandinsky6_mlx.utils.fixtures import save_fixture

TEXT_COMPONENTS = ("text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def find_component_root(root: Path, component: str) -> Path:
    """The directory holding `component/config.json` under `root` (Kaggle mounts outputs at varying depths)."""
    if (root / component / "config.json").is_file():
        return root
    for config in sorted(root.rglob(f"{component}/config.json")):
        return config.parent.parent
    raise FileNotFoundError(f"no {component}/config.json under {root}")


def encode_prompt(
    text_checkpoint: Path, prompt: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Text embeddings for `prompt`, returned in float32 on the CPU. Qwen runs in `dtype`, CLIP in float32."""
    from diffusers import Kandinsky6TI2VAPipeline
    from transformers import CLIPTextModel, CLIPTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    root = find_component_root(text_checkpoint, "text_encoder")
    qwen_kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
    if device == "cuda":
        qwen_kwargs["device_map"] = "auto"  # spread over the visible GPUs (e.g. Kaggle 2x T4), CPU if needed
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(root / "text_encoder", **qwen_kwargs).eval()
    if device != "cuda":
        qwen = qwen.to(device)
    first_device = next(qwen.parameters()).device
    clip = CLIPTextModel.from_pretrained(root / "text_encoder_2", dtype=torch.float32).eval().to(first_device)
    others = inspect.signature(Kandinsky6TI2VAPipeline.__init__).parameters
    pipe = Kandinsky6TI2VAPipeline(
        **{name: None for name in others if name not in ("self", *TEXT_COMPONENTS)},
        text_encoder=qwen,
        tokenizer=Qwen2_5_VLProcessor.from_pretrained(root / "tokenizer"),
        text_encoder_2=clip,
        tokenizer_2=CLIPTokenizer.from_pretrained(root / "tokenizer_2"),
    )
    with torch.no_grad():
        embeds, pooled, mask, *_ = pipe.encode_prompt(
            prompt, do_classifier_free_guidance=False, device=first_device, dtype=torch.float32
        )
    text = {"encoder_hidden_states": embeds.float().cpu(), "pooled_projections": pooled.float().cpu()}
    if mask is not None:
        text["encoder_attention_mask"] = mask.cpu()
    del pipe, qwen, clip
    gc.collect()
    return text


def main(argv: list[str] | None = None) -> int:
    from reference.recorder import library_versions

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16", help="Qwen dtype")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    text_prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text()
    started = time.time()
    text = encode_prompt(args.text_checkpoint, text_prompt, args.device, DTYPES[args.dtype])
    seconds = time.time() - started
    save_fixture(
        args.out,
        {f"inputs.{k}": v for k, v in text.items()},
        {
            "fixture": args.out.name,
            "prompt": text_prompt,
            "qwen_dtype": args.dtype,
            "device": args.device,
            "seconds": round(seconds, 1),
            "versions": library_versions() | {"torch": torch.__version__},
        },
    )
    shapes = {k: list(v.shape) for k, v in text.items()}
    print(f"encoded in {seconds:.0f} s: {shapes} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
