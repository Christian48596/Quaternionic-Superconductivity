# High-definition generator for:
#   • Fig. 1: BdG quasiparticle gap map Egap(kx,ky)
#   • Fig. 2: Quaternion/Quartet vortex maps |Q(r)| and arg Q(r)
#
# This script is intended for paper-quality figure regeneration:
#   - Consistent typography across the manuscript figures (STIX fonts, LaTeX-like math)
#   - High-definition PNG output (600 dpi) + vector PDF output
#   - CSV dumps of the underlying data for reproducibility
#
# Units / conventions:
#   - Fig. 1: the plotted excitation energy is E_gap/t; the CSV stores E_gap.
#   - Fig. 2: spatial coordinates are lattice units x/a, y/a with a=1 in the plots
#            (the internal grid is an N×N lattice of points)
#
# Outputs created in the current directory:
#   Fig. 1:
#     - fig1_bdg_gapmap_nt_hd.png
#     - fig1_bdg_gapmap_nt_hd.pdf
#     - fig1_bdg_gapmap_nt_hd.csv     (columns: kx, ky, Egap)
#   Fig. 2:
#     - fig2_Q_mag_nt_cb_hd.png/.pdf  (amplitude panel)
#     - fig2_Q_phase_nt_cb_hd.png/.pdf (phase panel)
#     - fig2_Q_mag_nt_cb_hd.csv       (columns: x, y, Q_mag; x,y are grid indices)
#     - fig2_Q_phase_nt_cb_hd.csv     (columns: x, y, Q_phase; x,y are grid indices)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

# ------------------------- Global plot style (PRB-friendly) -------------------------
# Keep all figures visually consistent (fonts, tick sizes, etc.).
# NOTE: Heatmaps typically look cleaner without a background grid, so we disable axes.grid.
plt.rcParams.update({
    # Output resolution for raster figures. (PDF output is vector.)
    "savefig.dpi": 600,

    # Fonts: LaTeX-like math without requiring a full LaTeX install.
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",

    # Global typography scale tuned for PRB two-column layouts.
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,

    # Axis styling (clean, journal-like)
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,

    # Heatmap-style figures: grid usually distracts, so default off.
    "axes.grid": False,
})

# Output directory (current working directory by default)
outdir = Path(".")

# =============================================================================
# Figure 1: BdG gap map Egap(kx,ky)
# =============================================================================
def bulk_bdg(kx, ky, t=1.0, mu=-2.0, alpha=0.6, Delta_s=0.25, Delta_t=0.35):
    """Return the full 4×4 BdG matrix in the Kramers-hole basis.

    Momentum arrays broadcast; the last two output axes are Nambu × spin.
    For the real, time-reversal-invariant parameters used here,
    H' = [[h(k), M(k)], [M(k), -h(k)]], with
    h = xi I + alpha (sin(ky) sigma_x - sin(kx) sigma_y),
    M = Delta_s I + Delta_t (sin(kx) sigma_x + sin(ky) sigma_y).
    The conventional pairing matrix is Delta = M i sigma_y.
    """
    kx, ky = np.broadcast_arrays(kx, ky)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    xi = -2.0 * t * (np.cos(kx) + np.cos(ky)) - mu
    h = (xi[..., None, None] * np.eye(2)
         + alpha * (np.sin(ky)[..., None, None] * sx
                    - np.sin(kx)[..., None, None] * sy))
    pairing = (Delta_s * np.eye(2)
               + Delta_t * (np.sin(kx)[..., None, None] * sx
                            + np.sin(ky)[..., None, None] * sy))
    H = np.empty(kx.shape + (4, 4), dtype=complex)
    H[..., :2, :2] = h
    H[..., :2, 2:] = pairing
    H[..., 2:, :2] = pairing
    H[..., 2:, 2:] = -h
    return H


def make_fig1_bdg_gap_hd(
    Nk: int = 501,
    t: float = 1.0,
    mu: float = -2.0,
    alpha: float = 0.6,
    Delta_s: float = 0.25,
    Delta_t: float = 0.35,
):
    """
    Generate Fig. 1 from the eigenvalues of the full spinful BdG matrix.

    Egap(k) = min_n |E_n(k)| includes SOC and mixed-pairing branch splitting.

    Notes:
    - The colorbar shows Egap/t; t must be positive.
    - The CSV stores the full mesh (kx, ky, Egap) flattened row-wise.
    - The image extent is [-pi, pi] × [-pi, pi] with equal aspect ratio.

    Parameters
    ----------
    Nk : int
        Resolution along each momentum axis (Nk×Nk points). Higher Nk ⇒ smoother map, larger CSV.
    t, mu, alpha, Delta_s, Delta_t : float
        Real model parameters controlling the dispersion, SOC, and pairing.

    Outputs
    -------
    fig1_bdg_gapmap_nt_hd.png, fig1_bdg_gapmap_nt_hd.pdf, fig1_bdg_gapmap_nt_hd.csv
    """
    if t <= 0:
        raise ValueError("t must be positive for the E_gap/t color scale")
    # Build the k-grid
    kx = np.linspace(-np.pi, np.pi, Nk)
    ky = np.linspace(-np.pi, np.pi, Nk)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")

    energies = np.linalg.eigvalsh(bulk_bdg(KX, KY, t, mu, alpha, Delta_s, Delta_t))
    Egap = np.min(np.abs(energies), axis=-1)

    # ------------------------- Save raw data to CSV -------------------------
    # This makes the figure fully reproducible and allows downstream re-plotting.
    pd.DataFrame({
        "kx": KX.ravel(),
        "ky": KY.ravel(),
        "Egap": Egap.ravel()
    }).to_csv(outdir / "fig1_bdg_gapmap_nt_hd.csv", index=False)

    # ------------------------- Plot -------------------------
    fig = plt.figure(figsize=(7.6, 6.6))  # scales well in PRB after LaTeX inclusion
    ax = fig.add_subplot(111)

    im = ax.imshow(
        Egap / t,
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        origin="lower",
        interpolation="none",
    )

    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_aspect("equal")

    # Colorbar: keep readable and consistent with other figures
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$E_{\rm gap}/t$")
    cbar.ax.tick_params(labelsize=16)

    # Save as PNG (raster) and PDF (vector)
    fig.savefig(outdir / "fig1_bdg_gapmap_nt_hd.png", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outdir / "fig1_bdg_gapmap_nt_hd.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Fig. 1: minimum E_gap/t on the {Nk}×{Nk} mesh = {Egap.min() / t:.8f}")


# =============================================================================
# Helper: vortex magnitude and phase fields on a square grid
# =============================================================================
def _vortex_fields(
    N: int = 256,
    center=(0.50, 0.56),
    Q0: float = 0.42,
    xi: float = 8.0,
    a: float = 1.0,
):
    """
    Construct a simple vortex texture for a complex quartet order parameter Q(r):

        Q(r) = |Q(r)| exp[i arg Q(r)]

    with:
        |Q(r)| = Q0 * tanh( r / (sqrt(2) * xi) )
        arg Q(r) = atan2(dy, dx)

    Coordinates:
    - The computation is done on an integer grid (xx,yy) = 0..N-1
    - Physical/lattice coordinates are:
          x = (xx - x0) * a
          y = (yy - y0) * a
      so plotting x/a and y/a is natural; when a=1, x/a == x in grid units.

    Parameters
    ----------
    N : int
        Grid size (N×N).
    center : (float, float)
        Vortex center given as fractions of the grid (0..1).
        Example center=(0.50,0.56) means slightly above the center.
    Q0 : float
        Bulk amplitude far from the core (sets the amplitude scale; "arb. units").
    xi : float
        Core size parameter (in the same length units as a; with a=1, xi is in lattice spacings).
    a : float
        Lattice spacing. Axis labels use x/a, y/a; setting a=1 is standard.

    Returns
    -------
    Q_mag : (N,N) array
        Amplitude |Q(r)| (arb. units).
    Q_phase : (N,N) array
        Phase arg Q(r) in radians, in [-pi, pi].
    """
    # Grid indices
    yy, xx = np.mgrid[0:N, 0:N]

    # Convert fractional center to index coordinates (0..N-1)
    x0 = center[0] * (N - 1)
    y0 = center[1] * (N - 1)

    # Displacements in physical units
    dx = (xx - x0) * a
    dy = (yy - y0) * a
    r = np.hypot(dx, dy)

    # Vortex profile: tanh recovery of amplitude + polar-angle phase winding
    Q_mag = Q0 * np.tanh(r / (np.sqrt(2.0) * xi))
    Q_phase = np.arctan2(dy, dx)  # radians

    return Q_mag, Q_phase


# =============================================================================
# Figure 2: Vortex amplitude and phase maps
# =============================================================================
def make_fig2_vortex_maps_hd(
    N: int = 256,
    center=(0.50, 0.56),
    Q0: float = 0.42,
    xi: float = 8.0,
    a: float = 1.0,
    cmap_mag=None,
    cmap_phase=None,
):
    """
    Generate Fig. 2:
      (a) amplitude map |Q(r)|
      (b) phase map arg Q(r)

    Plot/units:
    - Axes are labeled as x/a and y/a. With a=1, these equal the grid coordinate indices.
    - Amplitude |Q| is in arbitrary/model units set by Q0.
    - Phase is in radians with a colorbar spanning [-pi, pi].

    Data export:
    - CSV stores x,y as integer grid indices (0..N-1), plus Q_mag or Q_phase.
      If you need physical coordinates, interpret x_phys = x*a, y_phys = y*a.

    Outputs
    -------
    fig2_Q_mag_nt_cb_hd.png/.pdf + fig2_Q_mag_nt_cb_hd.csv
    fig2_Q_phase_nt_cb_hd.png/.pdf + fig2_Q_phase_nt_cb_hd.csv
    """
    # Generate vortex fields
    Q_mag, Q_phase = _vortex_fields(N=N, center=tuple(center), Q0=Q0, xi=xi, a=a)

    # ------------------------- Save raw data -------------------------
    yy, xx = np.mgrid[0:N, 0:N]
    pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "Q_mag": Q_mag.ravel()}).to_csv(
        outdir / "fig2_Q_mag_nt_cb_hd.csv", index=False
    )
    pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "Q_phase": Q_phase.ravel()}).to_csv(
        outdir / "fig2_Q_phase_nt_cb_hd.csv", index=False
    )

    # Image extent in physical units (0..(N-1)*a). Since axes are x/a, y/a, this is natural.
    L = (N - 1) * a
    extent = (0.0, L, 0.0, L)

    # ---------- Panel A: |Q(r)| ----------
    figA = plt.figure(figsize=(7.4, 6.4))
    axA = figA.add_subplot(111)

    imA = axA.imshow(
        Q_mag,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        cmap=cmap_mag,
    )

    axA.set_xlabel(r"$x/a$")
    axA.set_ylabel(r"$y/a$")
    axA.set_aspect("equal")

    # “Nice” tick locations (good for N=256 and a=1); adjust if you change N or a
    axA.set_xticks([0, L/4, L/2, 3*L/4, L])
    axA.set_yticks([0, L/4, L/2, 3*L/4, L])

    cbarA = figA.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
    cbarA.set_label(r"$|Q(\mathbf{r})|$ (arb.~units)")
    cbarA.ax.tick_params(labelsize=16)

    figA.savefig(outdir / "fig2_Q_mag_nt_cb_hd.png", bbox_inches="tight", pad_inches=0.02)
    figA.savefig(outdir / "fig2_Q_mag_nt_cb_hd.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figA)

    # ---------- Panel B: arg Q(r) ----------
    figB = plt.figure(figsize=(7.4, 6.4))
    axB = figB.add_subplot(111)

    imB = axB.imshow(
        Q_phase,
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        cmap=cmap_phase,
    )

    axB.set_xlabel(r"$x/a$")
    axB.set_ylabel(r"$y/a$")
    axB.set_aspect("equal")
    axB.set_xticks([0, L/4, L/2, 3*L/4, L])
    axB.set_yticks([0, L/4, L/2, 3*L/4, L])

    cbarB = figB.colorbar(
        imB,
        ax=axB,
        fraction=0.046,
        pad=0.04,
        ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    )
    cbarB.ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    cbarB.set_label(r"$\arg Q(\mathbf{r})$ (rad)")
    cbarB.ax.tick_params(labelsize=16)

    figB.savefig(outdir / "fig2_Q_phase_nt_cb_hd.png", bbox_inches="tight", pad_inches=0.02)
    figB.savefig(outdir / "fig2_Q_phase_nt_cb_hd.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figB)


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the BdG gap and prescribed vortex maps.")
    parser.add_argument("--figure", choices=("1", "2", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    outdir = args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.figure in ("1", "both"):
        make_fig1_bdg_gap_hd()
    if args.figure in ("2", "both"):
        make_fig2_vortex_maps_hd(N=256, center=(0.50, 0.56), Q0=0.42, xi=8.0, a=1.0)
