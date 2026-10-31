"""Encode the Smoke prompt into Kandinsky 6 text embeddings on a Kaggle GPU (specs 05 and 09 stage B).

Input attached to the kernel: the output of `applehawk/k6-lite-diffusers-weights-text` (Qwen2.5-VL `text_encoder/`,
Lite-distill revision 4c487e8b). Internet is needed to clone github.com/vladislav-vasilenko/kandinsky6-mlx at COMMIT,
install the reference pins and download the small `tokenizer/`, `text_encoder_2/` (CLIP-L) and `tokenizer_2/`
folders of the same revision. Qwen runs in float16 on the GPU(s) with `device_map="auto"`, as the study notebook did on
T4s; Qwen in bf16 on a Kaggle CPU did not finish in 80 minutes.

Output in /kaggle/working: `smoke_text.{safetensors,json}` (1024 padded tokens x 3584, the CLIP pooled embedding and
the padding mask); it is attached to `kaggle/record_dit/` and copied to `fixtures/real/` on the Mac.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/vladislav-vasilenko/kandinsky6-mlx"
COMMIT = "c3504d2d03ea777f8ddc224f10e88fd898bbc630"
SOURCE_REPO = "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers"
REVISION = "4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81"
PINS = ["diffusers==0.42.1", "transformers==5.22.1", "accelerate==1.13.0"]
TEXT_EXTRA = ["tokenizer/*", "text_encoder_2/*", "tokenizer_2/*"]
SRC = Path("/tmp/kandinsky6-mlx")
TEXT_ROOT = Path("/tmp/k6-text")
OUT = Path("/kaggle/working")


def run(*args: str, cwd: Path | None = None) -> None:
    print("$", " ".join(args[:6]), "..." if len(args) > 6 else "", flush=True)
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
    return TEXT_ROOT


def main() -> None:
    started = time.time()
    run("git", "clone", "--quiet", REPO, str(SRC))
    run("git", "checkout", "--quiet", COMMIT, cwd=SRC)
    run(sys.executable, "-m", "pip", "install", "--quiet", *PINS)
    run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")
    run(
        sys.executable,
        str(SRC / "reference" / "encode_text.py"),
        "--text-checkpoint",
        str(text_root()),
        "--prompt-file",
        str(SRC / "fixtures" / "real_inputs" / "smoke_prompt.txt"),
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--out",
        str(OUT / "smoke_text"),
        cwd=SRC,
    )
    print(f"done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
