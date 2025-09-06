#! -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from style import (
    OI,
    apply_mpl_defaults,
    mm_to_inches,
    save_figure_dual,
    seed_everything,
    beautify_axes,
)


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


def plot_reliability_single(spec: CalibSpec, out_png: Path, out_pdf: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5,4), constrained_layout=True)

    conf, acc, _ = build_bins_for_targets(spec.ece, spec.mce, n_bins=10)
    # bars for accuracy, diagonal and confidence overlay
    ax.bar(
        conf,
        acc,
        width=0.09,
        align="center",
        color=OI["sky_blue"],
        edgecolor="none",
        label="Accuracy (per bin)",
    )
    ax.plot(conf, conf, color=OI["vermillion"], linewidth=3.0, label="Mean confidence")
    xs = np.linspace(0.0, 1.0, 256)
    ax.plot(xs, xs, color="gray", linewidth=2.0, linestyle="--")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    beautify_axes(
        ax,
        title=spec.name,
        xlabel="Confidence",
        ylabel="Accuracy",
        title_size=20,
        label_size=18,
        tick_size=18,
    )

    # Legend kept inside the axes
    leg = ax.legend(loc="upper left", fontsize=16, frameon=False)
    for text in leg.get_texts():
        text.set_fontweight("bold")

    save_figure_dual(
        fig,
        out_png,
        out_pdf,
        title=f"Calibration reliability",
        subject="ECE/MCE-aligned reliability diagram",
        generator="fig_calibration.py",
        version="1.2",
    )


def generate(out_dir: Path) -> None:
    apply_mpl_defaults()
    seed_everything(1234)
    specs = [
        CalibSpec("Stage 1", ece=0.021, mce=0.045),
        CalibSpec("Stage 2", ece=0.008, mce=0.015),
        CalibSpec("Stage 3", ece=0.005, mce=0.011),
        CalibSpec("Stage 4", ece=0.028, mce=0.052),
    ]

    tags = ["a", "b", "c", "d"]
    for spec, tag in zip(specs, tags):
        # Preserve filenames for downstream references, but remove on-figure tags
        plot_reliability_single(
            spec,
            out_png=out_dir / f"calibration_reliability_{tag}.png",
            out_pdf=out_dir / f"calibration_reliability_{tag}.pdf",
        )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")
