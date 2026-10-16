#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional strip diagnostic (legacy Fig. S1): spectrum of the BdG Hamiltonian (open along y, periodic along x)

This script constructs a 2D BdG model on a *strip* geometry:
  - periodic boundary conditions along x  → kx remains a good quantum number
  - open boundary conditions along y      → Ny sites ("strip width") in real space

For each kx, we build the BdG Hamiltonian H_BdG(kx) in Nambu space and diagonalize it.
The resulting eigenvalues E_n(kx) are plotted as a band-structure style spectrum.

This is intended as a publication-quality, reproducible figure generator:
  • consistent STIX typography across all figures in the repo (Figs. 1–5 + SM)
  • high-definition PNG (600 dpi) + vector PDF export
  • CSV dump containing the full E_n(kx) dataset for re-plotting

--------------------------------------------------------------------------------
Outputs
--------------------------------------------------------------------------------
Figure files:
  - figS1_edge_spectrum_nt_exact_hd.png
  - figS1_edge_spectrum_nt_exact_hd.pdf

Raw data:
  - figS1_edge_spectrum_nt_exact_hd.csv
    columns:
      kx, E_0000, E_0001, ..., E_(4Ny-1)
    where the E-columns are the sorted BdG eigenvalues at each kx.

--------------------------------------------------------------------------------
Units / conventions
--------------------------------------------------------------------------------
- Momentum kx is in lattice units (dimensionless), plotted over [-π, π].
- Energy is in "model units" set by the hopping t.
  The plot shows E/t; the CSV stores the full, unclipped energies E.

--------------------------------------------------------------------------------
How to run
--------------------------------------------------------------------------------
From the repository root:
  python make_figS1_edge_spectrum_nt.py --out-dir Figures

The default parameters are topologically trivial (nu=0); this spectrum is not
evidence for a protected helical edge crossing.

Runtime notes:
- The BdG matrix dimension is 4*Ny (spin × Nambu × Ny).
- Increasing Ny increases the number of subbands (and CPU cost).
- Increasing Nkx increases momentum resolution (and CPU cost).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

# =============================================================================
# Global plot style (match Figs. 1–5)
# =============================================================================
# STIX fonts give a LaTeX-like appearance without requiring a LaTeX backend.
# We keep the same typography across the entire repository for consistency.
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
    Save the figure in two formats:
      - PNG at high DPI for raster export
      - PDF for vector-quality inclusion in LaTeX/APS manuscripts

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure handle to save.
    base : str
        Output prefix (extension is appended automatically).
    """
    fig.savefig(f"{base}.png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# Spin matrices; the strip uses (Nambu, y, spin) ordering.
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
s0 = np.eye(2, dtype=complex)


def build_blocks(kx, Ny=60, t=1.0, mu=-2.0, alpha=0.6, Delta_s=0.25, Delta_t=0.35):
    """
    Construct electron block H_e(kx) and pairing block Δ(kx) for the strip BdG Hamiltonian.

    Structure:
      - H_e(kx) acts on (y ⊗ spin) space, dimension 2*Ny
      - Δ(kx) acts on the same space and couples particle/hole sectors in Nambu space

    Electron part:
      H_e = kinetic_x(kx) + kinetic_y + SOC_x(kx) + SOC_y
    with:
      kinetic_x ~ -2t cos(kx) - mu
      kinetic_y ~ -t (Ty_plus + Ty_minus)  (open chain hopping)
      SOC_x     ~ -alpha sin(kx) σ_y
      SOC_y     ~  alpha sin(ky) σ_x  → implemented via S_op

    Pairing:
      Δ = [ ψ I + d·σ ] i σ_y,
      with ψ = Δ_s and triplet d = Δ_t (sin kx, sin ky, 0).
      sin ky is represented by S_op on the strip.

    Returns
    -------
    H_e : (2Ny, 2Ny) complex ndarray
        Electron Hamiltonian block.
    Delta : (2Ny, 2Ny) complex ndarray
        Pairing block.
    """
    if Ny < 1:
        raise ValueError("Ny must be positive")
    Ty_plus = np.diag(np.ones(Ny - 1), k=1).astype(complex)
    Ty_minus = Ty_plus.T
    S_op = (Ty_plus - Ty_minus) / (2j)
    Iy = np.eye(Ny, dtype=complex)

    # ---------------- Kinetic energy ----------------
    # Onsite term includes dispersion along x (via cos kx) and chemical potential.
    h_onsite = (-2.0 * t * np.cos(kx) - mu) * np.kron(Iy, s0)

    # Nearest-neighbor hopping along y (open boundaries).
    h_yhop = -t * np.kron(Ty_plus + Ty_minus, s0)

    # ---------------- Spin–orbit coupling (Rashba-like) ----------------
    # Along y: sin(ky) σx, where sin(ky) becomes S_op on an open chain.
    h_soc_y = alpha * np.kron(S_op, sx)

    # Along x: -sin(kx) σy, with kx still a good quantum number.
    h_soc_x = -alpha * np.sin(kx) * np.kron(Iy, sy)

    H_e = h_onsite + h_yhop + h_soc_y + h_soc_x

    # ---------------- Pairing block ----------------
    # Triplet d-vector components in this model:
    #   d_x ∝ sin(kx), d_y ∝ sin(ky) (strip operator), d_z = 0
    d_x = Delta_t * np.sin(kx) * np.kron(Iy, sx)
    d_y = Delta_t * np.kron(S_op, sy)

    # A = ψ I + d·σ
    A = Delta_s * np.kron(Iy, s0) + d_x + d_y

    # Multiply by i σ_y (spin singlet structure) on the right.
    Delta = A @ np.kron(Iy, 1j * sy)

    return H_e, Delta

def strip_bdg(kx, Ny=60, t=1.0, mu=-2.0, alpha=0.6, Delta_s=0.25, Delta_t=0.35):
    """BdG matrix in the conventional (c_kx, c^dagger_-kx) Nambu basis."""
    H_e, Delta = build_blocks(kx, Ny, t, mu, alpha, Delta_s, Delta_t)
    H_e_minus, _ = build_blocks(-kx, Ny, t, mu, alpha, Delta_s, Delta_t)
    # Momentum reversal is essential in the hole block for SOC.
    return np.block([[H_e, Delta], [Delta.conj().T, -H_e_minus.T]])


def strip_spectrum(Ny=60, Nkx=101, t=1.0, mu=-2.0, alpha=0.6, Delta_s=0.25, Delta_t=0.35):
    """Return momenta and all sorted eigenvalues, without clipping."""
    kx_grid = np.linspace(-np.pi, np.pi, Nkx)
    eigvals_all = np.empty((Nkx, 4 * Ny), dtype=float)
    for i, kx in enumerate(kx_grid):
        eigvals_all[i] = np.linalg.eigvalsh(
            strip_bdg(kx, Ny, t, mu, alpha, Delta_s, Delta_t)
        )
    return kx_grid, eigvals_all


def make_strip_spectrum(outdir=Path("."), Ny=60, Nkx=101):
    """Export the default t=1 strip spectrum and its plot."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kx_grid, eigvals_all = strip_spectrum(Ny=Ny, Nkx=Nkx)
    cols = [f"E_{i:04d}" for i in range(eigvals_all.shape[1])]
    df = pd.DataFrame(eigvals_all, columns=cols)
    df.insert(0, "kx", kx_grid)
    base = outdir / "figS1_edge_spectrum_nt_exact_hd"
    df.to_csv(f"{base}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for n in range(eigvals_all.shape[1]):
        ax.plot(kx_grid, eigvals_all[:, n], linewidth=0.9, alpha=0.95)
    ax.set_xlim(-np.pi, np.pi)
    # Restrict the display only; retain the full bands in the CSV.
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel(r"$k_x$", fontsize=30)
    ax.set_ylabel(r"$E/t$", fontsize=30)
    ax.set_xticks([-np.pi, -2, -1, 0, 1, 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-2$", r"$-1$", r"$0$", r"$1$", r"$2$", r"$\pi$"])
    ax.set_yticks(np.linspace(-2, 2, 9))
    ax.grid(True, which="major", linestyle=(0, (3, 3)), alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=(0, (3, 3)), alpha=0.25)
    ax.tick_params(direction="out")
    save_hd(fig, base)
    print(f"Wrote: {base}.(png/pdf/csv)")
    print(f"Minimum |E|/t on the strip mesh = {np.min(np.abs(eigvals_all)):.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the optional BdG strip diagnostic.")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    make_strip_spectrum(outdir=args.out_dir)
