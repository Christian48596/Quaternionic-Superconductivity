#!/usr/bin/env python3
"""
Test script: single-particle dispersion for the two-orbital chain.

This does NOT use TeNPy; it just constructs the 2x2 orbital Hamiltonian
for each k and prints/plots the two bands:

  E_±(k) = ε(k) ± t_perp,  with  ε(k) = -2 t cos(k) - μ.
"""

import numpy as np
import matplotlib.pyplot as plt


def band_energies(k, t, t_perp, mu):
    """Return the two band energies E_minus(k), E_plus(k)."""
    eps = -2.0 * t * np.cos(k) - mu
    E_plus = eps + t_perp
    E_minus = eps - t_perp
    return E_minus, E_plus


def main():
    # --- choose parameters consistent with your DMRG ---
    t = 1.0
    t_perp = 0.6
    mu = -1.2

    # k-grid in the 1D Brillouin zone [-pi, pi)
    Nk = 201
    k_vals = np.linspace(-np.pi, np.pi, Nk, endpoint=False)

    E_minus_vals = []
    E_plus_vals = []

    for k in k_vals:
        Emin, Emax = band_energies(k, t, t_perp, mu)
        E_minus_vals.append(Emin)
        E_plus_vals.append(Emax)

    E_minus_vals = np.array(E_minus_vals)
    E_plus_vals = np.array(E_plus_vals)

    # Print a few sample points for sanity check
    print("Sample dispersion values:")
    for k_test in [0.0, np.pi / 2, np.pi]:
        Emin, Emax = band_energies(k_test, t, t_perp, mu)
        print(f"k = {k_test:6.3f}  ->  E_minus = {Emin:8.4f},  E_plus = {Emax:8.4f}")

    # Optional: plot the two bands
    plt.figure(figsize=(4.0, 3.0))
    plt.plot(k_vals, E_minus_vals, label=r"$E_-(k)$")
    plt.plot(k_vals, E_plus_vals, label=r"$E_+(k)$")
    plt.axhline(0.0, color="k", linewidth=0.5)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)$")
    plt.title("Two-orbital chain dispersion")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("test_quartet_dispersion.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
