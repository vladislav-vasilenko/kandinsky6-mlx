"""Kaggle reference kernels clone a pinned commit that must exist and hold, unchanged, what the kernel runs."""

import ast
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Per kernel: directory, code file, files run from the pinned checkout, attached Kaggle outputs.
KERNELS = {
    "text": (
        "kaggle/encode_text",
        "encode_text_k6.py",
        (
            "reference/encode_text.py",
            "reference/recorder.py",
            "src/kandinsky6_mlx/utils/fixtures.py",
            "fixtures/real_inputs/smoke_prompt.txt",
        ),
        {"applehawk/k6-lite-diffusers-weights-text"},
    ),
    "dit": (
        "kaggle/record_dit",
        "record_dit_k6.py",
        (
            "reference/record_dit_real.py",
            "reference/encode_text.py",
            "reference/recorder.py",
            "src/kandinsky6_mlx/utils/fixtures.py",
        ),
        {"applehawk/k6-lite-diffusers-weights-media", "applehawk/k6-lite-smoke-text"},
    ),
    "vae": (
        "kaggle/record_vae",
        "record_vae_k6.py",
        (
            "reference/record_video_real.py",
            "reference/recorder.py",
            "src/kandinsky6_mlx/utils/fixtures.py",
            "src/kandinsky6_mlx/convert/layouts.py",
        ),
        {"applehawk/k6-lite-diffusers-weights-media"},
    ),
}


def constant(kernel: str, name: str):
    directory, code_file, *_ = KERNELS[kernel]
    tree = ast.parse((ROOT / directory / code_file).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=False, cwd=ROOT, capture_output=True, text=True)


def require_commit(commit: str) -> None:
    if git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        shallow = git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"
        hint = "the clone is shallow; fetch the full history (CI: actions/checkout fetch-depth: 0)" if shallow else ""
        raise AssertionError(f"pinned commit {commit} is not in this repository. {hint}")


@pytest.mark.parametrize("kernel", KERNELS)
def test_pinned_commit_is_in_history_and_has_the_recorder(kernel):
    commit = constant(kernel, "COMMIT")
    require_commit(commit)
    assert git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0
    for path in KERNELS[kernel][2]:
        assert git("cat-file", "-e", f"{commit}:{path}").returncode == 0, path


@pytest.mark.parametrize("kernel", KERNELS)
def test_recorder_at_pinned_commit_matches_head(kernel):
    """The kernel runs the pinned recorder; if any file it runs changed since, the pin must move."""
    commit = constant(kernel, "COMMIT")
    require_commit(commit)
    changed = git("diff", "--name-only", commit, "HEAD", "--", *KERNELS[kernel][2]).stdout.split()
    assert not changed, f"changed since the pinned commit: {changed}"


@pytest.mark.parametrize("kernel", KERNELS)
def test_metadata_attaches_weights_and_needs_internet(kernel):
    directory, code_file, _, sources = KERNELS[kernel]
    meta = json.loads((ROOT / directory / "kernel-metadata.json").read_text())
    assert meta["code_file"] == code_file and meta["enable_internet"]
    # Only text encoding needs a GPU; the float32 references run on the CPU.
    assert meta["enable_gpu"] == (kernel == "text")
    assert set(meta["kernel_sources"]) == sources
