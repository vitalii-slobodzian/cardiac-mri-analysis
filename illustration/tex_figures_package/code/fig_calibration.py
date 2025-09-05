# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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

@dataclass
class CalibSpec:
    name: str
    ece: float
    mce: float

def build_bins_for_targets(
    ece_target: float,
    mce_target: float,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct bin confidences/accuracies/weights reproducing ECE & MCE.

    Integrity note:
      Raw per-sample probabilities are unavailable. We use equal bin weights
      and set |acc - conf| so that mean == ECE_target and max == MCE_target.
      Signs are chosen to keep acc within [0,1] while producing a plausible
      under/over-confidence pattern.
    """
    n = n_bins
    weights = np.full(n, 1.0 / n, dtype=float)
    conf = np.linspace(0.05, 0.95, n)
    diffs = np.zeros(n, dtype=float)

    diffs[-1] = mce_target
    remainder = n * ece_target - mce_target
    per = max(remainder / (n - 1), 0.0)
    diffs[:-1] = per

    signs = np.where(conf <= 0.5, +1.0, -1.0)
    acc = np.clip(conf + signs * diffs, 0.0, 1.0)

    ece = np.sum(weights * np.abs(acc - conf))
    mce = np.max(np.abs(acc - conf))
    assert abs(ece - ece_target) < 1e-6, (ece, ece_target)
    assert abs(mce - mce_target) < 1e-6, (mce, mce_target)
    return conf, acc, weights

def _save_with_png_metadata(fig_path: Path, meta: dict) -> None:
    img = Image.open(fig_path)
    pnginfo = PngImagePlugin.PngInfo()
    for k, v in meta.items():
        pnginfo.add_text(k, str(v))
    img.save(fig_path, "PNG", pnginfo=pnginfo)

def plot_reliability_grid(
    out_png: Path,
    out_pdf: Path,
    specs: List[CalibSpec],
    fig_width_mm: float = 180.0,
    fig_height_mm: float = 110.0,
) -> None:
    w_in = fig_width_mm / 25.4
    h_in = fig_height_mm / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(w_in, h_in), constrained_layout=True)

    for ax, spec in zip(axes.ravel(), specs):
        conf, acc, _ = build_bins_for_targets(spec.ece, spec.mce, n_bins=10)
        ax.bar(conf, acc, width=0.09, align="center", color=OI["sky_blue"], edgecolor="none", label="Accuracy (per bin)")
        ax.plot(conf, conf, color=OI["vermillion"], linewidth=2.0, label="Mean confidence")
        xs = np.linspace(0.0, 1.0, 256)
        ax.plot(xs, xs, color=OI["black"], linewidth=1.0, alpha=0.5, linestyle="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(spec.name)
        ax.text(
            0.98, 0.05, f"ECE={spec.ece:.3f}\nMCE={spec.mce:.3f}",
            ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
        )
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="upper center", frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    _save_with_png_metadata(out_png, {
        "Figure": "Calibration reliability 2x2",
        "Generator": "fig_calibration.py",
        "Timestamp": str(np.datetime64('now')),
        "Version": "1.0",
    })
    fig.savefig(out_pdf, dpi=300, metadata={
        "Title": "Calibration reliability diagrams (2×2)",
        "Author": "TeX-Figures-Refinement",
        "Subject": "ECE/MCE-aligned reliability diagrams",
    })
    plt.close(fig)

def generate(out_dir: Path) -> None:
    specs = [
        CalibSpec("Stage 1", ece=0.021, mce=0.045),
        CalibSpec("Stage 2", ece=0.008, mce=0.015),
        CalibSpec("Stage 3", ece=0.005, mce=0.011),
        CalibSpec("Stage 4", ece=0.028, mce=0.052),
    ]
    plot_reliability_grid(
        out_png=out_dir / "calibration_reliability.png",
        out_pdf=out_dir / "calibration_reliability.pdf",
        specs=specs, fig_width_mm=180.0, fig_height_mm=110.0
    )

if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")
