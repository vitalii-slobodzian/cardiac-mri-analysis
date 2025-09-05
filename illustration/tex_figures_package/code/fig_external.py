# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from style import OI, apply_mpl_defaults, mm_to_inches, save_figure_dual, seed_everything


def plot_external(out_png: Path, out_pdf: Path) -> None:
    cohorts = ["ACDC (test)", "M&Ms (external)"]
    metrics = ["LV", "RV", "MYO"]

    means = np.array([[0.957, 0.931, 0.908], [0.910, 0.860, 0.850]])
    lowers = np.array([[0.950, 0.920, 0.900], [0.900, 0.840, 0.830]])
    uppers = np.array([[0.960, 0.940, 0.910], [0.920, 0.880, 0.860]])

    # Validate CI consistency
    assert np.all(lowers <= means) and np.all(means <= uppers)

    err_minus = means - lowers
    err_plus = uppers - means

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=mm_to_inches(140, 95), constrained_layout=True)

    ax.bar(
        x - width / 2,
        means[0],
        width,
        yerr=[err_minus[0], err_plus[0]],
        capsize=3,
        label=cohorts[0],
        color=OI["blue"],
    )
    ax.bar(
        x + width / 2,
        means[1],
        width,
        yerr=[err_minus[1], err_plus[1]],
        capsize=3,
        label=cohorts[1],
        color=OI["orange"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Dice (mean ± 95% CI)")
    ax.set_title("External Validation: ACDC vs. M&Ms")
    ax.legend(loc="lower right")

    save_figure_dual(
        fig,
        out_png,
        out_pdf,
        title="External validation bars",
        subject="Dice with 95% CIs",
        generator="fig_external.py",
        version="1.1",
    )


def generate(out_dir: Path) -> None:
    apply_mpl_defaults()
    seed_everything(1234)
    plot_external(out_dir / "external_validation_bars.png", out_dir / "external_validation_bars.pdf")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")

