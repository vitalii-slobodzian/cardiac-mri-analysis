
"""
generate_all_figures.py
CLI entry point to regenerate all figures.
"""
from __future__ import annotations
from pathlib import Path
import argparse
from fig_calibration import generate_calibration
from fig_dice import generate_dice
from fig_roc import generate_roc
from fig_confusion import generate_multiclass, generate_cascade
from fig_sota import generate_sota
from fig_external import generate_external

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="figures", help="Output directory for figures")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_calibration(out_dir)
    generate_dice(out_dir)
    generate_roc(out_dir)
    generate_multiclass(out_dir)
    generate_cascade(out_dir)
    generate_sota(out_dir)
    generate_external(out_dir)

if __name__ == "__main__":
    main()
