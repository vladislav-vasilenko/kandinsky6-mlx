"""The Kaggle reference kernel of spec 05 clones a pinned commit that must exist and hold what the kernel runs."""

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "kaggle" / "record_dit"


def constant(name: str):
    tree = ast.parse((KERNEL / "record_dit_k6.py").read_text())
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


def test_pinned_commit_is_in_history_and_has_the_recorder():
    commit = constant("COMMIT")
    require_commit(commit)
    assert git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0
    for path in ("reference/record_dit_real.py", "reference/recorder.py", "fixtures/real_inputs/smoke_prompt.txt"):
        assert git("cat-file", "-e", f"{commit}:{path}").returncode == 0, path


def test_recorder_at_pinned_commit_matches_head():
    """The kernel runs the pinned recorder; if it changed since, the pin must move."""
    commit = constant("COMMIT")
    require_commit(commit)
    assert (
        git("diff", "--quiet", commit, "HEAD", "--", "reference/", "src/kandinsky6_mlx/utils/fixtures.py").returncode
        == 0
    )


def test_metadata_attaches_media_and_text_weights_and_needs_internet():
    meta = json.loads((KERNEL / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "record_dit_k6.py" and meta["enable_internet"] and not meta["enable_gpu"]
    assert set(meta["kernel_sources"]) == {
        "applehawk/k6-lite-diffusers-weights-media",
        "applehawk/k6-lite-diffusers-weights-text",
    }
