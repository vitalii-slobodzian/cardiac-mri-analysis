"""
Shared plotting style and utilities for figure generation.

Centralizes rcParams, color palette, sizing helpers, and saving with
consistent metadata across PNG/PDF/SVG outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, PngImagePlugin

# Okabe–Ito color palette
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


def apply_mpl_defaults() -> None:
    """Apply consistent Matplotlib style for publication-quality figures."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "savefig.dpi": 300,
            "figure.dpi": 110,
            "pdf.fonttype": 42,  # embed fonts
            "ps.fonttype": 42,
            "legend.frameon": False,
        }
    )


def mm_to_inches(w_mm: float, h_mm: float) -> Tuple[float, float]:
    return w_mm / 25.4, h_mm / 25.4


def seed_everything(seed: int = 1234) -> None:
    np.random.seed(seed)


def _add_png_metadata(path: Path, meta: dict) -> None:
    try:
        img = Image.open(path)
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            pnginfo.add_text(k, str(v))
        img.save(path, "PNG", pnginfo=pnginfo)
    except Exception:
        # Best-effort: metadata embedding is optional. Keep the saved image.
        pass


def save_figure_dual(
    fig: mpl.figure.Figure,
    out_png: Path,
    out_pdf: Path,
    *,
    title: Optional[str] = None,
    subject: Optional[str] = None,
    generator: str = "tex_figures_package",
    version: str = "1.0",
    also_svg: bool = True,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    _add_png_metadata(
        out_png,
        {
            "Figure": title or out_png.stem,
            "Generator": generator,
            "Timestamp": str(np.datetime64("now")),
            "Version": version,
        },
    )

    # Save PDF with metadata
    fig.savefig(
        out_pdf,
        dpi=300,
        bbox_inches="tight",
        metadata={
            "Title": title or out_pdf.stem,
            "Author": generator,
            "Subject": subject or "",
        },
    )

    # Optional SVG for vector workflows (helps match .ipynb outputs)
    if also_svg:
        fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")

