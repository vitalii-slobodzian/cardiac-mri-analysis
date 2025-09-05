# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

def _save_with_png_metadata(fig_path: Path, meta: dict) -> None:
    img = Image.open(fig_path)
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(k, str(v))
    img.save(fig_path, "PNG", pnginfo=pnginfo)

def plot_dice(out_png: Path, out_pdf: Path) -> None:
    experiments = np.arange(1, 6)
    data_ed = {
        "LV":  [0.911, 0.920, 0.919, 0.956, 0.974],
        "RV":  [0.842, 0.902, 0.892, 0.939, 0.947],
        "MYO": [0.812, 0.875, 0.855, 0.866, 0.896],
    }
    data_es = {
        "LV":  [0.890, 0.894, 0.887, 0.930, 0.940],
        "RV":  [0.871, 0.891, 0.873, 0.905, 0.915],
        "MYO": [0.832, 0.884, 0.885, 0.898, 0.920],
    }

    fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 100/25.4), constrained_layout=True)

    for ax, phase, data in zip(axes, ["ED", "ES"], [data_ed, data_es]):
        ax.plot(experiments, data["LV"], marker="o", color=OI["blue"], label="LV")
        ax.plot(experiments, data["RV"], marker="s", color=OI["vermillion"], label="RV")
        ax.plot(experiments, data["MYO"], marker="^", color=OI["bluish_green"], label="Myocardium")
        ax.set_xticks(experiments)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Dice coefficient")
        ax.set_title(f"{phase}")
        ax.set_ylim(0.78, 1.0)

    axes[0].legend(loc="lower right", frameon=False, ncols=3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    _save_with_png_metadata(out_png, {
        "Figure": "Dice comparison ED/ES",
        "Generator": "fig_dice.py",
        "Timestamp": str(np.datetime64('now')),
        "Version": "1.0",
    })
    fig.savefig(out_pdf, dpi=300, metadata={
        "Title": "Dice comparison across experiments (ED/ES)",
        "Author": "TeX-Figures-Refinement",
        "Subject": "Dice plots from manuscript table",
    })
    plt.close(fig)

def generate(out_dir: Path) -> None:
    plot_dice(out_dir / "dice_comparison_ed_es.png", out_dir / "dice_comparison_ed_es.pdf")

if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")
