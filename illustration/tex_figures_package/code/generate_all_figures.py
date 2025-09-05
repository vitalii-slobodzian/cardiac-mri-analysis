# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from fig_calibration import generate as gen_calib
from fig_dice import generate as gen_dice
from fig_confusion import generate as gen_cm
from fig_roc import generate as gen_roc
from fig_sota import generate as gen_sota
from fig_external import generate as gen_external

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for the wlscirep manuscript.")
    parser.add_argument(
        "--which", type=str, default="all",
        choices=["all", "calibration", "dice", "roc", "confusion", "sota", "external"],
        help="Which figures to generate."
    )
    parser.add_argument(
        "--outdir", type=str,
        default=str(Path(__file__).resolve().parents[1] / "figures"),
        help="Output directory for figures."
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(1234)

    if args.which in ("all", "calibration"):
        gen_calib(outdir)
    if args.which in ("all", "dice"):
        gen_dice(outdir)
    if args.which in ("all", "roc"):
        gen_roc(outdir)
    if args.which in ("all", "confusion"):
        gen_cm(outdir)
    if args.which in ("all", "sota"):
        gen_sota(outdir)
    if args.which in ("all", "external"):
        gen_external(outdir)

if __name__ == "__main__":
    main()
