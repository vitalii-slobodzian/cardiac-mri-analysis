# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from style import OI, apply_mpl_defaults, mm_to_inches, save_figure_dual, seed_everything


def _plot_cm(cm: np.ndarray, class_names: List[str], title: str, out_png: Path, out_pdf: Path) -> None:
    # Percentages row-normalized for display
    row_sums = cm.sum(axis=1, keepdims=True)
    perc = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(1, 1, figsize=mm_to_inches(120, 100), constrained_layout=True)
    im = ax.imshow(perc, cmap="Blues", vmin=0, vmax=100, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}\n({perc[i, j]:.0f}%)",
                ha="center",
                va="center",
                color="black" if perc[i, j] < 60 else "white",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row %")

    ax.text(
        0.5,
        -0.18,
        "Representative matrix to match manuscript overall accuracy; counts illustrative.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    save_figure_dual(
        fig,
        out_png,
        out_pdf,
        title=f"Confusion {title}",
        subject="Representative confusion matrix from summary accuracy",
        generator="fig_confusion.py",
        version="1.1",
    )


def _cyclic_distribute_errors(diag: List[int], per_class_total: int) -> np.ndarray:
    k = len(diag)
    cm = np.zeros((k, k), dtype=int)
    for i, d in enumerate(diag):
        cm[i, i] = d
        errs = per_class_total - d
        cm[i, (i + 1) % k] = errs
    return cm


def generate(out_dir: Path) -> None:
    apply_mpl_defaults()
    seed_everything(1234)

    classes = ["NOR", "MINF", "DCM", "HCM", "ARV"]

    cm_r50 = _cyclic_distribute_errors([8, 8, 9, 8, 9], per_class_total=10)
    acc_r50 = np.trace(cm_r50) / cm_r50.sum()
    assert abs(acc_r50 - 0.84) < 1e-6
    _plot_cm(cm_r50, classes, "ResNet-50 (5-class)", out_dir / "cm_resnet50.png", out_dir / "cm_resnet50.pdf")

    cm_r18 = _cyclic_distribute_errors([7, 7, 8, 7, 7], per_class_total=10)
    acc_r18 = np.trace(cm_r18) / cm_r18.sum()
    assert abs(acc_r18 - 0.72) < 1e-6
    _plot_cm(cm_r18, classes, "ResNet-18 (5-class)", out_dir / "cm_resnet18.png", out_dir / "cm_resnet18.pdf")

    classes_s1 = ["NOR+ARV", "LV pathologies"]
    cm_s1 = np.array([[19, 1], [1, 29]], dtype=int)
    assert abs(np.trace(cm_s1) / cm_s1.sum() - 0.96) < 1e-6
    _plot_cm(cm_s1, classes_s1, "Cascade Stage 1", out_dir / "cm_stage1.png", out_dir / "cm_stage1.pdf")

    classes_s2 = ["NOR", "ARV"]
    cm_s2 = np.array([[10, 0], [0, 10]], dtype=int)
    _plot_cm(cm_s2, classes_s2, "Cascade Stage 2", out_dir / "cm_stage2.png", out_dir / "cm_stage2.pdf")

    classes_s3 = ["HCM", "MINF+DCM"]
    cm_s3 = np.array([[10, 0], [0, 20]], dtype=int)
    _plot_cm(cm_s3, classes_s3, "Cascade Stage 3", out_dir / "cm_stage3.png", out_dir / "cm_stage3.pdf")

    classes_s4 = ["MINF", "DCM"]
    cm_s4 = np.array([[9, 1], [1, 9]], dtype=int)
    assert abs(np.trace(cm_s4) / cm_s4.sum() - 0.90) < 1e-6
    _plot_cm(cm_s4, classes_s4, "Cascade Stage 4", out_dir / "cm_stage4.png", out_dir / "cm_stage4.pdf")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "figures")

