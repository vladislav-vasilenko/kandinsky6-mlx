"""Generate the Torch MPS vs MLX example clips of the model card, with time and memory per clip.

Each clip is one cell of scripts/benchmark.py (a fresh process, the study notebook's configuration, the same text
embeddings, noise and image on both sides), so these are single cold runs: fit for showing the clips, not for the
statistics, which come from the benchmark matrix.

    uv run --extra text --group reference python scripts/generate_comparison.py --out docs/hf/lite-distill-mlx/examples
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

CLIPS = {
    "orbit_woman": ("orbit_woman", None),
    "fox": ("fox", None),
    "strange_tram": ("strange_tram", None),
    "mechanical_bloom": ("mechanical_bloom", None),
    "room_assembly": ("room_assembly", None),
    "race_driver": ("race_driver", None),
    "skydive_pov": ("skydive_pov", None),
    "aerial_silk": ("aerial_silk", None),
    "i2v_woman": ("orbit_woman", "fixtures/real_inputs/i2v_input.png"),
    "i2v_tram": ("strange_tram", "fixtures/real_inputs/i2v_input_tram.png"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs/hf/lite-distill-mlx/examples")
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/m3max-48gb-examples")
    parser.add_argument("--qwen-weights", type=Path, nargs="+", default=[benchmark.QWEN_DEFAULT])
    parser.add_argument("--preset", default="smoke", choices=sorted(benchmark.SIZES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", help="clip names")
    parser.add_argument("--impl", choices=("mlx", "diffusers"), help="one implementation only (default: both)")
    args = parser.parse_args(argv)

    prompts = json.loads((args.out / "prompts.json").read_text())
    text_dir = ROOT / "fixtures/real/examples"
    text_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or list(CLIPS)

    def prompt_key(name: str) -> str:
        """The preset's own long prompt at Baseline size, when there is one."""
        key = CLIPS[name][0]
        return f"{key}_baseline" if args.preset == "baseline" and f"{key}_baseline" in prompts else key

    for prompt_name in sorted({prompt_key(name) for name in names}):
        if not (text_dir / f"{prompt_name}.safetensors").is_file():
            command = [sys.executable, str(ROOT / "scripts/encode_prompt.py"), "--prompt", prompts[prompt_name]]
            command += ["--out", str(text_dir / prompt_name), "--qwen-weights", *map(str, args.qwen_weights)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "machine.json").write_text(json.dumps(benchmark.machine(), indent=2) + "\n")
    failed = 0
    for name in names:
        prompt_name, image = prompt_key(name), CLIPS[name][1]
        for impl, label, device in (("mlx", "mlx", "gpu"), ("diffusers", "torch_mps", "mps")):
            if args.impl not in (None, impl):
                continue
            mp4 = args.out / f"{name}_{args.preset}_seed{args.seed}_{label}.mp4"
            cell = benchmark.pipeline_cell(
                "examples",
                args.preset,
                "distill",
                f"{name}-{args.preset}-{label}",
                impl,
                device,
                text=str(text_dir / prompt_name),
                image=image,
                seed=args.seed,
                repeat=0,
                save_mp4=str(mp4),
            )
            record = benchmark.run_cell(cell, args.results / "logs" / f"{name}_{args.preset}_{label}.log") | {
                "round": 0
            }
            with (args.results / "results.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(benchmark.summary_line(record), flush=True)
            failed += record["status"] != "ok"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
