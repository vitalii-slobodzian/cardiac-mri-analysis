# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from style import OI, apply_mpl_defaults, mm_to_inches, save_figure_dual, seed_everything


def plot_sota(out_png: Path, out_pdf: Path) -> None:
    methods = [
        "Hu 2023",
        "da Silva 2024",
        "Ammar 2021",
        "Bourfiss 2023",
        "Hasan & Linte 2020",
        "Zhang 2022",
        "Benameur 2025*",
        "Ours",
    ]
    ed_lv = [0.968, 0.963, 0.964, 0.959, 0.963, 0.976, 0.978, 0.974]
    ed_rv = [0.946, 0.932, 0.935, 0.929, 0.924, 0.949, 0.945, 0.947]
    ed_myo = [0.902, 0.892, 0.889, 0.875, 0.901, 0.903, 0.905, 0.896]
    es_lv = [0.931, 0.911, 0.917, 0.921, 0.929, 0.950, 0.948, 0.940]
    es_rv = [0.899, 0.883, 0.879, 0.885, 0.887, 0.916, 0.919, 0.915]
    es_myo = [0.919, 0.901, 0.898, 0.895, 0.913, 0.918, 0.917, 0.920]

    # Basic value sanity
    for series in [ed_lv, ed_rv, ed_myo, es_lv, es_rv, es_myo]:
        assert all(0.0 <= v <= 1.0 for v in series)

    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=mm_to_inches(180, 110), constrained_layout=True)

    axes[0].plot(x, ed_lv, marker="o", color=OI["blue"], linewidth=2.0, label="LV")
    axes[0].plot(x, ed_rv, marker="s", color=OI["vermillion"], linewidth=2.0, label="RV")
    axes[0].plot(x, ed_myo, marker="^", color=OI["bluish_green"], linewidth=2.0, label="Myocardium")
    axes[0].set_title("ED")
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha="right")
    axes[0].set_ylabel("Dice")

    axes[1].plot(x, es_lv, marker="o", color=OI["blue"], linewidth=2.0, label="LV")
    axes[1].plot(x, es_rv, marker="s", color=OI["vermillion"], linewidth=2.0, label="RV")
    axes[1].plot(x, es_myo, marker="^", color=OI["bluish_green"], linewidth=2.0, label="Myocardium")
    axes[1].set_title("ES")
    axes[1].set_ylim(0.85, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha="right")

    axes[0].legend(loc="lower right", ncols=3)

    save_figure_dual(
        fig,
        out_png,
        out_pdf,
        title="SOTA Dice comparison",
        subject="Dice across methods and phases",
        generator="fig_sota.py",
        version="1.1",
    )


def generate(out_dir: Path) -> None:
    apply_mpl_defaults()
    seed_everything(1234)
    plot_sota(out_dir / "sota_dice_comparison.png", out_dir / "sota_dice_comparison.pdf")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")

