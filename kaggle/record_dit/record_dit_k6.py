"""Record the float32 reference of the Lite-distill DiT at Smoke size for spec 05 stage B (Kaggle CPU kernel).

Inputs attached to the kernel: the outputs of `applehawk/k6-lite-diffusers-weights-media` (transformer, Lite-distill
revision 4c487e8b) and `applehawk/k6-lite-smoke-text` (`smoke_text`, encoded on a GPU by `kaggle/encode_text/`).
Internet is needed to clone github.com/vladislav-vasilenko/kandinsky6-mlx at COMMIT and install the reference pins.
Only the float32 reference is recorded here (`--skip-bf16`); the bf16 tolerance basis is added later with
`record_dit_real.py --drift-only` on a GPU or Apple MPS, because bf16 on this CPU is extremely slow.

Output in /kaggle/working: `dit_smoke.{safetensors,json}` (about 0.8 GB); copy it, with `smoke_text.*` from the text
kernel, to `fixtures/real/` on the Mac. The float32 DiT needs about 15 GB of the session's 30 GB of RAM.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/vladislav-vasilenko/kandinsky6-mlx"
COMMIT = "c3504d2d03ea777f8ddc224f10e88fd898bbc630"
PINS = ["diffusers==0.42.1", "transformers==5.22.1", "accelerate==1.13.0"]
SRC = Path("/tmp/kandinsky6-mlx")
OUT = Path("/kaggle/working")


def run(*args: str, cwd: Path | None = None) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=cwd)


def main() -> None:
    started = time.time()
    run("git", "clone", "--quiet", REPO, str(SRC))
    run("git", "checkout", "--quiet", COMMIT, cwd=SRC)
    run(sys.executable, "-m", "pip", "install", "--quiet", *PINS)
    text = next(Path("/kaggle/input").rglob("smoke_text.safetensors"))
    run(
        sys.executable,
        str(SRC / "reference" / "record_dit_real.py"),
        "--checkpoint",
        "/kaggle/input",
        "--text-embeds",
        str(text),
        "--skip-bf16",
        "--out",
        str(OUT / "dit_smoke"),
        cwd=SRC,
    )
    sizes = {p.name: p.stat().st_size for p in sorted(OUT.glob("*.safetensors"))}
    print(f"done in {time.time() - started:.0f} s: {sizes}", flush=True)


if __name__ == "__main__":
    main()
