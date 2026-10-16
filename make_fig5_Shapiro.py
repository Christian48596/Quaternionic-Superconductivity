"""
Fig. 5: Shapiro step schematic (2e vs 4e ladder)

This script generates a *schematic* figure illustrating the expected Shapiro step
voltages at fixed microwave drive frequency f.

Physics background (as used in the manuscript text)
---------------------------------------------------
In a conventional Josephson junction (2e transport), Shapiro steps occur at:
    V_n = n (h f / 2e)

In a quartet-dominant regime (effective 4e transport / dominant second harmonic),
the Josephson frequency doubles and the steps shift to:
    V_n = n (h f / 4e)

If we normalize the voltage by (h f / 2e), the two ladders appear as:
  - 2e ladder: V/(h f / 2e) = n          → spacing 1
  - 4e ladder: V/(h f / 2e) = n/2        → spacing 1/2 (twice as many steps)

What is plotted here
--------------------
- X-axis: normalized voltage V/(h f / 2e), dimensionless.
- Y-axis: *categorical* (schematic), not a physical variable:
    • top row represents the 4e ladder  (denser steps)
    • bottom row represents the 2e ladder (sparser steps)
- Each “step” is drawn as a thick vertical segment.

Outputs
-------
Figures (high definition PNG + vector PDF):
  - fig5_Shapiro_steps_nt.png
  - fig5_Shapiro_steps_nt.pdf

Raw schematic geometry (CSV):
  - fig5_Shapiro_steps_nt.csv
    columns: series, x, y0, y1
    where x is the step position in normalized voltage units, and y0/y1 define the segment.

How to run
----------
python fig5_shapiro_steps.py

(or whatever filename you save this script as)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Global plot style (must match the rest of the repository)
# =============================================================================
# We use STIX fonts so that all manuscript figures share the same typography.
plt.rcParams.update({
    "savefig.dpi": 600,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",

    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,

    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

def save_hd(fig, base):
    """
    Save the figure as both:
      - PNG (600 dpi) for high-resolution raster output
      - PDF (vector) for LaTeX/APS inclusion
    """
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# =============================================================================
# Layout parameters for the schematic
# =============================================================================
# We define two horizontal "bands" (top = 4e, bottom = 2e) and draw vertical ticks in each.
x_min, x_max = -0.5, 7.8    # x-range shown (dimensionless: V/(hf/2e))
y_min, y_max = 0.0, 1.0     # total figure vertical range (schematic)
y_mid = 0.50                # dashed separator line between the two regimes

# Vertical spans of the step “ticks”
top_y0, top_y1 = 0.55, 0.95   # top band (4e ladder)
bot_y0, bot_y1 = 0.05, 0.45   # bottom band (2e ladder)

# Step positions in normalized voltage units:
# - 4e ladder has half spacing -> positions every 0.5
# - 2e ladder has integer spacing -> positions every 1.0
x_top = np.arange(0.0, 8.0, 0.5)   # 0, 0.5, ..., 7.5
x_bot = np.arange(0.0, 8.0, 1.0)   # 0, 1, ..., 7

# Visual style choices (matched to the manuscript)
lw = 9.0
c_orange = (0.894, 0.486, 0.0)    # deep orange used for step marks
grid_color = "#d9d9d9"            # light gray for grid/guide lines

# =============================================================================
# Plot construction
# =============================================================================
fig, ax = plt.subplots(figsize=(13.65, 7.7))   # wide aspect (schematic style)
ax.set_facecolor("white")

# Draw vertical "comb" ticks for the 4e (top) and 2e (bottom) ladders
for x in x_top:
    ax.plot([x, x], [top_y0, top_y1], color=c_orange, lw=lw, solid_capstyle="butt")
for x in x_bot:
    ax.plot([x, x], [bot_y0, bot_y1], color=c_orange, lw=lw, solid_capstyle="butt")

# Dashed separator between the two ladders (purely visual cue)
ax.hlines(
    y_mid, x_min, x_max,
    color=grid_color, linestyles=(0, (2, 6)), linewidth=1.5, zorder=0
)

# Vertical guide grid lines (only along x) to emphasize spacing
ax.set_axisbelow(True)
ax.grid(True, axis="x", linestyle="--", linewidth=1.0, alpha=0.45, color=grid_color)

# Axis limits and ticks
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xticks(np.arange(0, 8, 1))

# -----------------------------------------------------------------------------
# Axis labels (with local override for larger label fonts)
# -----------------------------------------------------------------------------
# X-axis: normalized voltage in dimensionless units
ax.set_xlabel(r"Voltage $V/(hf/2e)$", fontsize=28)

# Y-axis: categorical labeling (schematic), not a physical quantity.
# We place two tick labels at the centers of the top/bottom bands.
y_top_center = 0.5 * (top_y0 + top_y1)
y_bot_center = 0.5 * (bot_y0 + bot_y1)
ax.set_yticks([y_top_center, y_bot_center])
ax.set_yticklabels([r"$V_n=n\,hf/4e$", r"$V_n=n\,hf/2e$"], fontsize=28)

# Remove y tick marks (labels only; keeps schematic clean)
ax.tick_params(axis="y", length=0)

# Clean frame like PRB plots
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.25)
ax.spines["bottom"].set_linewidth(1.25)

plt.tight_layout()
save_hd(fig, "fig5_Shapiro_steps_nt")

# =============================================================================
# Save the schematic geometry to CSV
# =============================================================================
# This allows the figure to be reconstructed without the plotting code, and makes the
# positions of the 2e vs 4e steps explicit and reproducible.
rows = (
    [("4e_top", float(x), float(top_y0), float(top_y1)) for x in x_top] +
    [("2e_bot", float(x), float(bot_y0), float(bot_y1)) for x in x_bot]
)
pd.DataFrame(rows, columns=["series", "x", "y0", "y1"]).to_csv(
    "fig5_Shapiro_steps_nt.csv", index=False
)