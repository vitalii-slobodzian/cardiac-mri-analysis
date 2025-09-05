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

def plot_sota(out_png: Path, out_pdf: Path) -> None:
    methods = [
        "Hu 2023", "da Silva 2024", "Ammar 2021", "Bourfiss 2023",
        "Hasan & Linte 2020", "Zhang 2022", "Benameur 2025*", "Ours"
    ]
    ed_lv  = [0.968, 0.963, 0.964, 0.959, 0.963, 0.976, 0.978, 0.974]
    ed_rv  = [0.946, 0.932, 0.935, 0.929, 0.924, 0.949, 0.945, 0.947]
    ed_myo = [0.902, 0.892, 0.889, 0.875, 0.901, 0.903, 0.905, 0.896]
    es_lv  = [0.931, 0.911, 0.917, 0.921, 0.929, 0.950, 0.948, 0.940]
    es_rv  = [0.899, 0.883, 0.879, 0.885, 0.887, 0.916, 0.919, 0.915]
    es_myo = [0.919, 0.901, 0.898, 0.895, 0.913, 0.918, 0.917, 0.920]

    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(180/25.4, 110/25.4), constrained_layout=True)

    axes[0].plot(x, ed_lv,  marker="o", color=OI["blue"], label="LV")
    axes[0].plot(x, ed_rv,  marker="s", color=OI["vermillion"], label="RV")
    axes[0].plot(x, ed_myo, marker="^", color=OI["bluish_green"], label="Myocardium")
    axes[0].set_title("ED")
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha="right")
    axes[0].set_ylabel("Dice")

    axes[1].plot(x, es_lv,  marker="o", color=OI["blue"], label="LV")
    axes[1].plot(x, es_rv,  marker="s", color=OI["vermillion"], label="RV")
    axes[1].plot(x, es_myo, marker="^", color=OI["bluish_green"], label="Myocardium")
    axes[1].set_title("ES")
    axes[1].set_ylim(0.85, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha="right")

    axes[0].legend(loc="lower right", frameon=False, ncols=3)

    fig.savefig(out_png, dpi=300)
    _save_with_png_metadata(out_png, {
        "Figure": "SOTA Dice comparison",
        "Generator": "fig_sota.py",
        "Timestamp": str(np.datetime64('now')),
        "Version": "1.0",
    })
    fig.savefig(out_pdf, dpi=300, metadata={
        "Title": "SOTA Dice comparison",
        "Author": "TeX-Figures-Refinement",
        "Subject": "Dice across methods and phases",
    })
    plt.close(fig)

def generate(out_dir: Path) -> None:
    plot_sota(out_dir / "sota_dice_comparison.png", out_dir / "sota_dice_comparison.pdf")

if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")
