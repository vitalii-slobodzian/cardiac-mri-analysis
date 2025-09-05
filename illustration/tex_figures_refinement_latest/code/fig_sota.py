
"""
fig_sota.py
SOTA segmentation comparison chart using manuscript-provided Dice values.
Creates six single-panel plots (ED/ES × LV/RV/MYO) and composites them.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

METHODS = [
    "Hu 2023", "da Silva 2024", "Ammar 2021", "Bourfiss 2023",
    "Hasan & Linte 2020", "Zhang 2022", "Benameur 2025*", "Ours"
]

# Values copied from manuscript table
ED = {
    "LV":  [0.968, 0.963, 0.964, 0.959, 0.963, 0.976, 0.978, 0.974],
    "RV":  [0.946, 0.932, 0.935, 0.929, 0.924, 0.949, 0.945, 0.947],
    "Myo": [0.902, 0.892, 0.889, 0.875, 0.901, 0.903, 0.905, 0.896],
}
ES = {
    "LV":  [0.931, 0.911, 0.917, 0.921, 0.929, 0.950, 0.948, 0.940],
    "RV":  [0.899, 0.883, 0.879, 0.885, 0.887, 0.916, 0.919, 0.915],
    "Myo": [0.919, 0.901, 0.898, 0.895, 0.913, 0.918, 0.917, 0.920],
}

def _save(fig_path: Path) -> None:
    plt.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

def _panel(phase: str, struct: str, out: Path) -> Path:
    data = ED if phase == "ED" else ES
    vals = data[struct]
    x = np.arange(len(METHODS))
    plt.figure(figsize=(5.2, 2.8))
    ax = plt.gca()
    ax.plot(x, vals, marker="o", linewidth=2.0, color=OKABE_ITO["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=30, ha="right")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.86 if (phase == "ED" and struct == "Myo") else 0.87, 0.99)
    ax.set_title(f"{struct} — {phase}")
    _save(out)
    return out

def _composite(grid: list[Path], out_path: Path) -> None:
    # 6 images in 2 rows x 3 cols
    imgs = [Image.open(p.with_suffix(".png")) for p in grid]
    w, h = imgs[0].size
    cols, rows = 3, 2
    canvas = Image.new("RGB", (cols * w, rows * h), "white")
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas.paste(im, (c * w, r * h))
    canvas.save(out_path.with_suffix(".png"))
    canvas.save(out_path.with_suffix(".pdf"))

def generate_sota(out_dir: Path) -> None:
    panels = []
    for phase in ["ED", "ES"]:
        for struct in ["LV", "RV", "Myo"]:
            panels.append(_panel(phase, struct, out_dir / f"sota_{phase.lower()}_{struct.lower()}"))
    _composite(panels, out_dir / "sota_dice_comparison")
