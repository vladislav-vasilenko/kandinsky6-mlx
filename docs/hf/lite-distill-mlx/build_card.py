"""Build README.md of the Hugging Face model card from card.template.md, prompts and measured results.

uv run python docs/hf/lite-distill-mlx/build_card.py
"""

import html
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
BASE = "https://huggingface.co/VVladTech/Kandinsky-6.0-Lite-distill-5s-MLX/resolve/main/examples"
prompts = json.loads((HERE / "examples" / "prompts.json").read_text())


def video(stem: str, label: str) -> str:
    url = f"{BASE}/{stem}.mp4"
    return (
        f'<video controls playsinline preload="metadata" width="100%" poster="{BASE}/{stem}-poster.webp" '
        f'aria-label="{label}"><source src="{url}" type="video/mp4"><a href="{url}">{label} MP4</a></video>'
    )


def pair(name: str, input_image: str | None = None, preset: str = "smoke") -> str:
    columns = []
    if input_image:
        columns.append(("Input image", f'<img src="{BASE}/{input_image}" alt="Input image" width="100%">'))
    for label, suffix in (("Torch MPS (diffusers)", "torch_mps"), ("MLX", "mlx")):
        columns.append((label, video(f"{name}_{preset}_seed42_{suffix}", f"{label}: {name}")))
    width = 100 // len(columns)
    head = "".join(f'<th width="{width}%">{label}</th>' for label, _ in columns)
    body = "".join(f'<td width="{width}%">{cell}</td>' for _, cell in columns)
    return f'<table width="100%">\n<thead><tr>{head}</tr></thead>\n<tbody><tr>{body}</tr></tbody>\n</table>'


def prompt_block(key: str) -> str:
    return f"<details>\n<summary>Prompt</summary>\n\n{html.escape(prompts[key])}\n\n</details>"


records = {}
for line in (ROOT / "benchmarks/results/m3max-48gb-examples/results.jsonl").read_text().splitlines():
    record = json.loads(line)
    records[record["cell"]["id"].split("/")[-1]] = record  # the last run of a clip wins


def medians(names: list[str], label: str) -> dict:
    runs = [records[f"{n}-smoke-{label}"] for n in names]
    stage = lambda s: statistics.median(r["runs"][0]["seconds"].get(s, 0.0) for r in runs)
    total = lambda r, load: sum(v for k, v in r["runs"][0]["seconds"].items() if load or k != "load")
    return {
        "load": stage("load"),
        "image": stage("image_encode"),
        "denoise": stage("denoise"),
        "video": stage("video_decode"),
        "audio": stage("audio_decode"),
        "total": statistics.median(total(r, True) for r in runs),
        "no_load": statistics.median(total(r, False) for r in runs),
        "memory": statistics.median(r["footprint_peak_gib"] for r in runs),
    }


T2V = ["orbit_woman", "fox", "strange_tram", "mechanical_bloom", "room_assembly"]
I2V = ["i2v_woman", "i2v_tram"]


def perf_rows(names: list[str], title: str) -> str:
    rows = []
    for label, name in (("torch_mps", "Torch MPS (diffusers)"), ("mlx", "MLX")):
        m = medians(names, label)
        rows.append(
            f"| {title} | {name} | {m['load']:.1f} | {m['denoise']:.1f} | {m['video']:.1f} | {m['audio']:.1f} | "
            f"**{m['total']:.1f}** | {m['no_load']:.1f} | {m['memory']:.1f} |"
        )
    return "\n".join(rows)


def memory_rows() -> str:
    """Median over the text-to-video clips of the peak OS footprint in each stage."""

    def stage(label: str, name: str) -> float:
        values = [records[f"{clip}-smoke-{label}"]["runs"][0]["footprint_peak_gib"][name] for clip in T2V]
        return statistics.median(values)

    rows = []
    for name, title in (
        ("load", "Loading"),
        ("denoise", "Denoising"),
        ("video_decode", "Video decoding (the run's peak)"),
    ):
        torch_mps, mlx = stage("torch_mps", name), stage("mlx", name)
        rows.append(f"| {title} | {torch_mps:.1f} | {mlx:.1f} | {torch_mps - mlx:.1f} |")
    return "\n".join(rows)


text_rows = {}
results = ROOT / "benchmarks/results/m3max-48gb/results.jsonl"
for line in results.read_text().splitlines():
    record = json.loads(line)
    if record["cell"]["id"].startswith("text/") and record["status"] == "ok":
        text_rows.setdefault(record["cell"]["id"], []).append(record)


def text_row(cell: str, name: str) -> str:
    runs = text_rows[cell]
    warm = statistics.median(statistics.median(r["seconds"]["encode"] for r in x["runs"][1:]) for x in runs)
    memory = statistics.median(x["footprint_peak_gib"] for x in runs)
    return f"| {name} | {warm:.2f} | {memory:.1f} | {len(runs)} |"


replacements = {
    "{{PAIR_WOMAN}}": pair("orbit_woman"),
    "{{PROMPT_WOMAN}}": prompt_block("orbit_woman"),
    "{{PAIR_FOX}}": pair("fox"),
    "{{PROMPT_FOX}}": prompt_block("fox"),
    "{{PAIR_TRAM}}": pair("strange_tram"),
    "{{PROMPT_TRAM}}": prompt_block("strange_tram"),
    "{{PAIR_BLOOM}}": pair("mechanical_bloom"),
    "{{PROMPT_BLOOM}}": prompt_block("mechanical_bloom"),
    "{{PAIR_ROOM}}": pair("room_assembly"),
    "{{PROMPT_ROOM}}": prompt_block("room_assembly"),
    "{{PAIR_I2V_WOMAN}}": pair("i2v_woman", "i2v_input_woman.webp"),
    "{{PAIR_I2V_TRAM}}": pair("i2v_tram", "i2v_input_tram.webp"),
    "{{PAIR_TRAM_5S}}": pair("strange_tram", preset="baseline"),
    "{{PROMPT_TRAM_5S}}": prompt_block("strange_tram_baseline"),
    "{{PAIR_FOX_5S}}": pair("fox", preset="baseline"),
    "{{PROMPT_FOX_5S}}": prompt_block("fox_baseline"),
    "{{BASELINE_VIDEO}}": video("orbit_woman_baseline_seed42_mlx", "MLX: 5 s at 864x480"),
    "{{FRAME_SHEET}}": f"{BASE}/torch_mps_vs_mlx_frame12.webp",
    "{{MEMORY_STAGES}}": memory_rows(),
    "{{PERF_T2V}}": perf_rows(T2V, "Text-to-video, 5 prompts"),
    "{{PERF_I2V}}": perf_rows(I2V, "Image-to-video, 2 images"),
    "{{TEXT_MLX}}": text_row("text/mlx-bf16", "MLX, Qwen bf16 + CLIP"),
    "{{TEXT_8BIT}}": text_row("text/mlx-8bit", "MLX, Qwen 8-bit + CLIP"),
    "{{TEXT_TORCH}}": text_row("text/torch-mps-bf16", "Torch MPS, Qwen bf16 + CLIP"),
}
card = (HERE / "card.template.md").read_text()
for key, value in replacements.items():
    card = card.replace(key, value)
if "{{" in card:
    raise SystemExit("unfilled placeholder in the card")
(HERE / "README.md").write_text(card)
print("README.md", len(card), "bytes")
