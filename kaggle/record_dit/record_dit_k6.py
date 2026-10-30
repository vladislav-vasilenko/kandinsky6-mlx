"""Record the float32 reference of the Lite-distill DiT at Smoke size for spec 05 stage B (Kaggle CPU kernel).

Inputs attached to the kernel: the outputs of `applehawk/k6-lite-diffusers-weights-media` (transformer) and
`applehawk/k6-lite-diffusers-weights-text` (Qwen2.5-VL `text_encoder/`), both Lite-distill revision 4c487e8b.
Internet is needed: the kernel clones github.com/vladislav-vasilenko/kandinsky6-mlx at COMMIT, installs the reference
pins and downloads the small `tokenizer/`, `text_encoder_2/` (CLIP-L) and `tokenizer_2/` folders of the same revision.

Output in /kaggle/working: `dit_smoke.{safetensors,json}` (~0.8 GB) and `smoke_text.{safetensors,json}`; copy them
to `fixtures/real/` on the Mac. Memory peaks near 16 GB (Qwen in bf16), then ~15 GB (DiT in float32 plus the
recording); a Kaggle CPU session has 30 GB.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/vladislav-vasilenko/kandinsky6-mlx"
COMMIT = "b1c77deee348ec1c98d0a823a166d56c6f8658cc"
SOURCE_REPO = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"
REVISION = "4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81"
PINS = ["diffusers==0.42.1", "transformers==5.22.1", "accelerate==1.13.0"]
TEXT_EXTRA = ["tokenizer/*", "text_encoder_2/*", "tokenizer_2/*"]
SRC = Path("/tmp/kandinsky6-mlx")
TEXT_ROOT = Path("/tmp/k6-text")
OUT = Path("/kaggle/working")


def run(*args: str, cwd: Path | None = None) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=cwd)


def text_root() -> Path:
    """Qwen from the attached input plus the small text folders from the Hub, joined by symlinks in one root."""
    from huggingface_hub import snapshot_download

    qwen = next(Path("/kaggle/input").rglob("text_encoder/config.json")).parent
    extra = Path(snapshot_download(SOURCE_REPO, revision=REVISION, allow_patterns=TEXT_EXTRA, local_dir="/tmp/k6-extra"))
    TEXT_ROOT.mkdir(parents=True, exist_ok=True)
    (TEXT_ROOT / "text_encoder").symlink_to(qwen, target_is_directory=True)
    for name in ("tokenizer", "text_encoder_2", "tokenizer_2"):
        (TEXT_ROOT / name).symlink_to(extra / name, target_is_directory=True)
    print("text root:", {p.name: str(p.resolve()) for p in TEXT_ROOT.iterdir()}, flush=True)
    return TEXT_ROOT


def main() -> None:
    started = time.time()
    run("git", "clone", "--quiet", REPO, str(SRC))
    run("git", "checkout", "--quiet", COMMIT, cwd=SRC)
    run(sys.executable, "-m", "pip", "install", "--quiet", *PINS)
    prompt = (SRC / "fixtures" / "real_inputs" / "smoke_prompt.txt").read_text()
    run(
        sys.executable,
        str(SRC / "reference" / "record_dit_real.py"),
        "--checkpoint",
        "/kaggle/input",
        "--text-checkpoint",
        str(text_root()),
        "--prompt",
        prompt,
        "--save-text-embeds",
        str(OUT / "smoke_text"),
        "--out",
        str(OUT / "dit_smoke"),
        cwd=SRC,
    )
    sizes = {p.name: p.stat().st_size for p in sorted(OUT.glob("*.safetensors"))}
    print(f"done in {time.time() - started:.0f} s: {sizes}", flush=True)


if __name__ == "__main__":
    main()
