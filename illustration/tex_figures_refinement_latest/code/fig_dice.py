
"""
fig_dice.py
Plots ED and ES Dice comparisons for Experiments 1–5
(Values taken verbatim from the manuscript table.)
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

def _save(fig_path: Path) -> None:
    plt.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

def _composite(img_a: Path, img_b: Path, out_path: Path) -> None:
    A = Image.open(img_a.with_suffix(".png"))
    B = Image.open(img_b.with_suffix(".png"))
    w, h = A.size
    canvas = Image.new("RGB", (2 * w, h), "white")
    canvas.paste(A, (0, 0))
    canvas.paste(B, (w, 0))
    canvas.save(out_path.with_suffix(".png"))
    canvas.save(out_path.with_suffix(".pdf"))

def _plot_panel(values: dict, phase: str, out: Path) -> None:
    x = np.arange(1, 6)  # Experiments 1..5
    plt.figure(figsize=(3.4, 3.0))
    ax = plt.gca()
    ax.plot(x, values["LV"], marker="o", linewidth=2.0, color=OKABE_ITO["blue"], label="LV")
    ax.plot(x, values["RV"], marker="s", linewidth=2.0, color=OKABE_ITO["orange"], label="RV")
    ax.plot(x, values["Myo"], marker="^", linewidth=2.0, color=OKABE_ITO["green"], label="Myocardium")
    ax.set_xticks(x)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.80, 0.99 if phase == "ED" else 0.96)
    ax.set_title(f"{phase}")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _save(out)

def generate_dice(out_dir: Path) -> None:
    # ED
    ed = {
        "LV":  [0.911, 0.920, 0.919, 0.956, 0.974],
        "RV":  [0.842, 0.902, 0.892, 0.939, 0.947],
        "Myo": [0.812, 0.875, 0.855, 0.866, 0.896],
    }
    # ES
    es = {
        "LV":  [0.890, 0.894, 0.887, 0.930, 0.940],
        "RV":  [0.871, 0.891, 0.873, 0.905, 0.915],
        "Myo": [0.832, 0.884, 0.885, 0.898, 0.920],
    }

    ed_path = out_dir / "dice_comparison_ed"
    es_path = out_dir / "dice_comparison_es"
    _plot_panel(ed, "ED", ed_path)
    _plot_panel(es, "ES", es_path)

    _composite(ed_path, es_path, out_dir / "dice_comparison_ed_es")
