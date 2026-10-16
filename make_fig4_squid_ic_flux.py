"""
Fig. 4: SQUID critical-current modulation Ic(Φ) for different CPR regimes

This script generates a publication-quality SQUID interference pattern for a symmetric
two-junction SQUID (identical junctions) assuming a current–phase relation (CPR) of the form:

    I(φ) = I1 sin φ + I2 sin(2φ)

We compare three regimes:
  (1) 2e-dominant:  I1 ≠ 0, I2 = 0   (conventional Josephson CPR)
  (2) 4e-dominant:  I1 = 0, I2 ≠ 0   (dominant second harmonic → half flux periodicity)
  (3) Mixed:        I1 ≠ 0, I2 ≠ 0

Physics captured (schematically):
- The SQUID phase difference across the two junctions is shifted by ±δ with δ = π Φ/Φ0
  for a symmetric loop.
- The SQUID critical current is taken as the maximum magnitude of the total current
  over the free phase variable φ:
      Ic(Φ) = max_φ | I_total(φ, Φ) |
- Flux is plotted as Φ/Φ0, where Φ0 ≡ h/2e, so the x-axis is dimensionless.
- The plotted Ic is in arbitrary current units set by the chosen (I1, I2) amplitudes.

Outputs
-------
Figure files (high definition PNG + vector PDF):
  - fig4_SQUID_Ic_flux_nt_hd.png
  - fig4_SQUID_Ic_flux_nt_hd.pdf

Raw data (CSV):
  - fig4_SQUID_Ic_flux_nt_hd.csv
    columns: flux_Phi_over_Phi0, Ic_2e, Ic_4e, Ic_mixed

How to run
----------
From the repository root:
  python make_fig4_squid_ic_flux.py

(or whatever filename you save this script as)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Global plot style (keep consistent with the rest of the repository)
# =============================================================================
plt.rcParams.update({
    "savefig.dpi": 600,

    # Typography: STIX for consistent LaTeX-like appearance across all figures
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",

    # Sizes tuned for PRB two-column plots (can be rescaled in LaTeX)
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,

    # Clean axis aesthetics
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

def Ic_total(flux, I1, I2, n_phi=20001):
    """
    Compute SQUID critical current Ic(Φ) for a symmetric two-junction SQUID.

    Model:
    - Two identical junctions in a loop with flux Φ.
    - For a symmetric SQUID, the gauge-invariant phase differences across the junctions
      are shifted by ±δ, where:
          δ = π * (Φ/Φ0)
      with Φ0 ≡ h/2e.
    - Total current is the sum of the two junction currents:
          I_total(φ; Φ) = I(φ + δ) + I(φ - δ)
      where φ is a free (overall) phase that can adjust to maximize the current.

    CPR per junction:
        I(φ) = I1 sin φ + I2 sin(2φ)

    Critical current definition:
        Ic(Φ) = max_φ | I_total(φ; Φ) |

    Parameters
    ----------
    flux : float
        Normalized flux Φ/Φ0 (dimensionless).
    I1, I2 : float
        Harmonic amplitudes (arbitrary current units).
    n_phi : int
        Number of φ-samples used to approximate the max over φ.
        Larger n_phi gives a more accurate maximum (at higher cost).

    Returns
    -------
    float
        Ic(Φ) in arbitrary units set by (I1, I2).
    """
    # SQUID phase shift from the applied flux
    d = np.pi * flux

    # Sample φ densely to approximate max_φ |I_total|
    phi = np.linspace(0, 2*np.pi, n_phi)

    # Symmetric SQUID: currents add from φ+δ and φ−δ
    I = (
        I1 * (np.sin(phi + d) + np.sin(phi - d)) +
        I2 * (np.sin(2*(phi + d)) + np.sin(2*(phi - d)))
    )

    # Critical current is the maximum magnitude
    return np.max(np.abs(I))

def save_hd(fig, base):
    """
    Save the figure as:
      - PNG at 600 dpi for high-resolution raster output
      - PDF for vector-quality inclusion in LaTeX
    """
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Flux axis: sweep one full conventional flux quantum (0 → 1 in Φ/Φ0 units)
    # -------------------------------------------------------------------------
    flux = np.linspace(0.0, 1.0, 1001)

    # -------------------------------------------------------------------------
    # Three CPR regimes:
    #   - 2e-dominant: only sin φ harmonic
    #   - 4e-dominant: only sin(2φ) harmonic (period halving)
    #   - Mixed: both harmonics present
    # Note: All values are "arb. current units" (relative scales matter).
    # -------------------------------------------------------------------------
    Ic_2e  = np.array([Ic_total(f, I1=1.0, I2=0.0) for f in flux])
    Ic_4e  = np.array([Ic_total(f, I1=0.0, I2=1.0) for f in flux])
    Ic_mix = np.array([Ic_total(f, I1=1.0, I2=0.5) for f in flux])

    # -------------------------------------------------------------------------
    # Save raw data to CSV for reproducibility / re-plotting
    # -------------------------------------------------------------------------
    df = pd.DataFrame({
        "flux_Phi_over_Phi0": flux,
        "Ic_2e": Ic_2e,
        "Ic_4e": Ic_4e,
        "Ic_mixed": Ic_mix
    })
    df.to_csv("fig4_SQUID_Ic_flux_nt_hd.csv", index=False)

    # -------------------------------------------------------------------------
    # Plot: PRB-friendly layout (7.6×5.2 inches). Scale in LaTeX if desired.
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(7.6, 5.2))
    ax = fig.add_subplot(111)

    ax.plot(flux, Ic_2e,  lw=2.6, label=r"2e-dominant ($I_1$)")
    ax.plot(flux, Ic_4e,  lw=2.6, label=r"4e-dominant ($I_2$)")
    ax.plot(flux, Ic_mix, lw=2.6, label=r"Mixed ($I_1,I_2$)")

    # Axis limits chosen to match the intended panel framing
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 2.62)

    # Units:
    #  - Φ/Φ0 is dimensionless (Φ0=h/2e)
    #  - Ic is in arbitrary units set by I1, I2
    ax.set_xlabel(r"Flux $\Phi/\Phi_0$")
    ax.set_ylabel(r"$I_c(\Phi)$ (arb.~units)")

    # Light dashed grid for readability (optional but usually helpful)
    ax.grid(True, linestyle=(0, (3, 3)), alpha=0.35)
    ax.tick_params(direction="out")

    # Clean “PRB look”: remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: centered at top, no frame (matches manuscript aesthetic)
    ax.legend(loc="upper center", frameon=False)

    save_hd(fig, "fig4_SQUID_Ic_flux_nt_hd")