
"""
fig_external.py
ACDC vs M&Ms external validation bars with 95% CIs, using manuscript values.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

def _save(fig_path: Path) -> None:
    plt.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

def generate_external(out_dir: Path) -> None:
    labels = ["LV", "RV", "Myo"]
    x = np.arange(len(labels))
    width = 0.36

    # Means and CIs from manuscript table
    acdc_mean = np.array([0.957, 0.931, 0.908])
    acdc_lo   = np.array([0.950, 0.920, 0.900])
    acdc_hi   = np.array([0.960, 0.940, 0.910])
    mms_mean  = np.array([0.910, 0.860, 0.850])
    mms_lo    = np.array([0.900, 0.840, 0.830])
    mms_hi    = np.array([0.920, 0.880, 0.860])

    acdc_err = np.vstack([acdc_mean - acdc_lo, acdc_hi - acdc_mean])
    mms_err  = np.vstack([mms_mean - mms_lo,  mms_hi  - mms_mean])

    plt.figure(figsize=(4.2, 2.9))
    ax = plt.gca()
    ax.bar(x - width/2, acdc_mean, width, yerr=acdc_err, capsize=3, label="ACDC", color=OKABE_ITO["blue"])
    ax.bar(x + width/2, mms_mean,  width, yerr=mms_err,  capsize=3, label="M&Ms", color=OKABE_ITO["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dice (mean ± 95% CI)")
    ax.set_ylim(0.80, 0.99)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _save(out_dir / "external_validation_bars")
