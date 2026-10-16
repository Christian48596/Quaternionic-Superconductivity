"""
Fig. 3 (two panels): Josephson CPR and illustrative AC emission (2e vs 4e channel)

This script generates publication-quality (PRB/APS-friendly) figures with:
  • consistent STIX typography (matching other figures in the repository)
  • high-definition PNG output (600 dpi) + vector PDF output
  • CSV dumps of the plotted data for reproducibility

Outputs
-------
Figures (written to the repository root):
  - fig3_AC_emission_nt2_hd.png
  - fig3_AC_emission_nt2_hd.pdf
  - fig3_CPR_nt2_hd.png
  - fig3_CPR_nt2_hd.pdf

Data (written to ./csv/):
  - csv/fig3_AC_emission_nt2.csv   columns: t, I_2e, I_4e
  - csv/fig3_CPR_nt2.csv           columns: phi, I_total, I_2phi, I_1phi

Units / conventions
-------------------
Panel (AC emission):
  - Time is shown in arbitrary (dimensionless) units: in code we plot sin(t) and sin(2t).
    This is a schematic visualization of frequency doubling (f_J → 2 f_J) rather than a
    calibrated physical time axis.
  - Current is normalized (dimensionless), since sine waves are plotted with unit amplitude.

Panel (CPR):
  - Phase φ is in radians (dimensionless angle, plotted from 0 to 2π).
  - Current is in arbitrary/model units set by chosen harmonic amplitudes I1 and I2
    (here I1=0.12 and I2=1.0), i.e. it is not converted to amperes.

How to run
----------
From the repository root:
  python make_fig3_ac_and_cpr.py

(Or whatever filename you save this script as.)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Global plot style (must match the rest of the repo figures)
# =============================================================================
# We keep a single font/typography standard across all figures so that panels look
# uniform when placed together in the manuscript.
plt.rcParams.update({
    # High-resolution raster export; PDF is vector regardless of dpi.
    "savefig.dpi": 600,

    # STIX fonts mimic LaTeX output closely without requiring a LaTeX backend.
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",

    # Typographic scale tuned for PRB two-column inclusion (can be scaled in LaTeX).
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,

    # Clean axis style
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,

    # Default: no grid (enable per plot for better reading)
    "axes.grid": False,
})

# Directory for CSV data dumps (keeps repo root cleaner)
os.makedirs("csv", exist_ok=True)

def save_hd(fig, base):
    """
    Save a figure in two formats:
      1) PNG at high DPI for quick viewing / raster inclusion
      2) PDF for vector-quality manuscript inclusion

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure handle to save.
    base : str
        Output prefix (no extension). Extensions .png and .pdf are appended.
    """
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# =============================================================================
# Fig. 3A: AC emission schematic (frequency doubling)
# =============================================================================
# We use a dimensionless "time" variable t and plot sin(t) (2e channel) vs sin(2t) (4e channel).
# This shows a doubling of oscillation frequency, consistent with f_J → 2 f_J in the quartet regime.
t = np.linspace(0, 4*np.pi, 1600)
I_2e = np.sin(t)      # schematic 2e Josephson oscillation
I_4e = np.sin(2*t)    # schematic 4e oscillation (doubled frequency)

# Save raw data for reproducibility
pd.DataFrame({"t": t, "I_2e": I_2e, "I_4e": I_4e}).to_csv(
    "csv/fig3_AC_emission_nt2.csv", index=False
)

# Plot
fig = plt.figure(figsize=(7.6, 5.2))  # PRB-friendly aspect ratio
ax = fig.add_subplot(111)

ax.plot(t, I_2e, label="2e channel", linewidth=2.6)
ax.plot(t, I_4e, label="4e channel", linewidth=2.6)

# Axis labels: keep consistent with manuscript text
ax.set_xlabel(r"Time (arb.~units)")
ax.set_ylabel(r"Current (normalized)")

# Light dashed grid helps reading amplitudes without cluttering
ax.grid(True, linestyle=(0, (3, 3)), alpha=0.35)
ax.tick_params(direction="out")

# Legend in the upper-right to avoid overlapping curves
ax.legend(framealpha=1.0, loc="upper right")

save_hd(fig, "fig3_AC_emission_nt2_hd")

# =============================================================================
# Fig. 3B: CPR (current–phase relation) with dominant second harmonic
# =============================================================================
# CPR model used:
#   I(φ) = I1 sin φ + I2 sin(2φ)
# The "quartet-dominant" regime is illustrated by choosing I2 >> I1.
phi = np.linspace(0, 2*np.pi, 1600)
I1, I2 = 0.12, 1.0    # example parameters (arb. units); choose to emphasize I2 dominance

I_1phi  = I1 * np.sin(phi)       # first harmonic contribution (2e-like)
I_2phi  = I2 * np.sin(2*phi)     # second harmonic contribution (4e-like)
I_total = I_1phi + I_2phi        # total CPR

# Save raw data
pd.DataFrame({
    "phi": phi,
    "I_total": I_total,
    "I_2phi": I_2phi,
    "I_1phi": I_1phi
}).to_csv("csv/fig3_CPR_nt2.csv", index=False)

# Plot
fig = plt.figure(figsize=(7.6, 5.2))
ax = fig.add_subplot(111)

ax.plot(phi, I_total, label=r"$I(\varphi)=I_1\sin\varphi + I_2\sin 2\varphi$", linewidth=2.6)
ax.plot(phi, I_2phi,  label=r"$I_2\sin 2\varphi$", linestyle="--", linewidth=2.6)
ax.plot(phi, I_1phi,  label=r"$I_1\sin \varphi$", linestyle=":",  linewidth=2.6)

ax.set_xlabel(r"$\varphi$")                 # radians
ax.set_ylabel(r"Current (arb.~units)")      # scale set by I1, I2

ax.grid(True, linestyle=(0, (3, 3)), alpha=0.35)
ax.tick_params(direction="out")

# Legend placement:
# move slightly up/right so it does not cover the peak of I_total.
ax.legend(
    framealpha=0.95,
    loc="upper right",
    bbox_to_anchor=(1.02, 1.02),
    borderaxespad=0.0
)

save_hd(fig, "fig3_CPR_nt2_hd")