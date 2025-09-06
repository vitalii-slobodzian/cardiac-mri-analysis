# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from style import OI, apply_mpl_defaults, mm_to_inches, save_figure_dual, seed_everything, beautify_axes


def plot_dice(out_png: Path, out_pdf: Path) -> None:
    experiments = np.arange(1, 6)
    data_ed = {
        "LV": [0.911, 0.920, 0.919, 0.956, 0.974],
        "RV": [0.842, 0.902, 0.892, 0.939, 0.947],
        "MYO": [0.812, 0.875, 0.855, 0.866, 0.896],
    }
    data_es = {
        "LV": [0.890, 0.894, 0.887, 0.930, 0.940],
        "RV": [0.871, 0.891, 0.873, 0.905, 0.915],
        "MYO": [0.832, 0.884, 0.885, 0.898, 0.920],
    }

    # Basic value sanity
    for series in list(data_ed.values()) + list(data_es.values()):
        assert all(0.0 <= v <= 1.0 for v in series)

    fig, axes = plt.subplots(1, 2, figsize=mm_to_inches(180, 100), constrained_layout=True)

    for ax, phase, data in zip(axes, ["ED", "ES"], [data_ed, data_es]):
        ax.plot(experiments, data["LV"], marker="o", color=OI["blue"], linewidth=3.0, label="LV")
        ax.plot(experiments, data["RV"], marker="s", color=OI["vermillion"], linewidth=3.0, label="RV")
        ax.plot(experiments, data["MYO"], marker="^", color=OI["bluish_green"], linewidth=3.0, label="Myocardium")
        ax.set_xticks(experiments)
        beautify_axes(ax, title=f"{phase}", xlabel="Experiment", ylabel="Dice coefficient")
        ymin = min(min(data["LV"]), min(data["RV"]), min(data["MYO"]))
        ax.set_ylim(max(0.78, ymin - 0.05), 1.0)

    leg = axes[0].legend(loc="lower right", ncols=3, fontsize=16, frameon=False)
    for text in leg.get_texts():
        text.set_fontweight("bold")

    save_figure_dual(
        fig,
        out_png,
        out_pdf,
        title="Dice comparison ED/ES",
        subject="Dice plots",
        generator="fig_dice.py",
        version="1.1",
    )


def generate(out_dir: Path) -> None:
    apply_mpl_defaults()
    seed_everything(1234)
    plot_dice(out_dir / "dice_comparison_ed_es.png", out_dir / "dice_comparison_ed_es.pdf")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")

