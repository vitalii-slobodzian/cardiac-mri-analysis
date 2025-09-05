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

def plot_external(out_png: Path, out_pdf: Path) -> None:
    cohorts = ["ACDC (test)", "M&Ms (external)"]
    metrics = ["LV", "RV", "MYO"]

    means = np.array([[0.957, 0.931, 0.908],
                      [0.910, 0.860, 0.850]])
    lowers = np.array([[0.950, 0.920, 0.900],
                       [0.900, 0.840, 0.830]])
    uppers = np.array([[0.960, 0.940, 0.910],
                       [0.920, 0.880, 0.860]])

    err_minus = means - lowers
    err_plus = uppers - means

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(140/25.4, 95/25.4), constrained_layout=True)

    ax.bar(x - width/2, means[0], width, yerr=[err_minus[0], err_plus[0]],
           capsize=3, label=cohorts[0], color=OI["blue"])
    ax.bar(x + width/2, means[1], width, yerr=[err_minus[1], err_plus[1]],
           capsize=3, label=cohorts[1], color=OI["vermillion"])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title("External Validation: ACDC vs. M&Ms")
    ax.legend(loc="lower right", frameon=False)

    fig.savefig(out_png, dpi=300)
    _save_with_png_metadata(out_png, {
        "Figure": "External validation bars",
        "Generator": "fig_external.py",
        "Timestamp": str(np.datetime64('now')),
        "Version": "1.0",
    })
    fig.savefig(out_pdf, dpi=300, metadata={
        "Title": "External validation bars",
        "Author": "TeX-Figures-Refinement",
        "Subject": "Dice with 95% CIs",
    })
    plt.close(fig)

def generate(out_dir: Path) -> None:
    plot_external(out_dir / "external_validation_bars.png", out_dir / "external_validation_bars.pdf")

if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")
