"""Physical symmetry and analytic-limit checks for the figure Hamiltonians."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from make_fig1_fig2 import bulk_bdg
from make_figS1_edge_spectrum_nt import build_blocks, make_strip_spectrum, strip_bdg


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)


class BulkTests(unittest.TestCase):
    def test_hermiticity_particle_hole_and_time_reversal(self):
        C = np.kron(SY, SY)
        T = np.kron(np.eye(2), 1j * SY)
        for kx, ky in ((0.71, -0.43), (-1.2, 0.81), (0.0, np.pi)):
            with self.subTest(kx=kx, ky=ky):
                H, Hminus = bulk_bdg(kx, ky), bulk_bdg(-kx, -ky)
                assert_allclose(H, H.conj().T, atol=1e-13)
                assert_allclose(C @ H.conj() @ C.conj().T, -Hminus, atol=1e-13)
                assert_allclose(T @ H.conj() @ T.conj().T, Hminus, atol=1e-13)

    def test_conventional_nambu_basis_and_fermi_antisymmetry(self):
        kx, ky = 0.71, -0.43
        H, Hminus = bulk_bdg(kx, ky), bulk_bdg(-kx, -ky)
        Delta = H[:2, 2:] @ (1j * SY)
        Delta_minus = Hminus[:2, 2:] @ (1j * SY)
        assert_allclose(Delta, -Delta_minus.T, atol=1e-13)
        Hconventional = np.block([[H[:2, :2], Delta],
                                  [Delta.conj().T, -Hminus[:2, :2].T]])
        W = np.zeros((4, 4), dtype=complex)
        W[:2, :2], W[2:, 2:] = np.eye(2), 1j * SY
        assert_allclose(H, W @ Hconventional @ W.conj().T, atol=1e-13)

    def test_zero_soc_mixed_pairing_branches(self):
        kx = np.array([0.71, -1.2, 0.0])
        ky = np.array([-0.43, 0.81, np.pi])
        xi = -2 * (np.cos(kx) + np.cos(ky)) + 2
        d = 0.35 * np.hypot(np.sin(kx), np.sin(ky))
        Eplus = np.sqrt(xi**2 + (0.25 + d)**2)
        Eminus = np.sqrt(xi**2 + (0.25 - d)**2)
        expected = np.sort(np.stack((-Eplus, -Eminus, Eminus, Eplus), axis=-1))
        assert_allclose(np.linalg.eigvalsh(bulk_bdg(kx, ky, alpha=0)), expected, atol=1e-13)

    def test_pure_singlet_rashba_branches(self):
        kx, ky = 0.71, -0.43
        xi = -2 * (np.cos(kx) + np.cos(ky)) + 2
        for alpha in (0.0, 0.6, 1.0):
            with self.subTest(alpha=alpha):
                g = alpha * np.hypot(np.sin(kx), np.sin(ky))
                Eplus = np.sqrt((xi + g)**2 + 0.25**2)
                Eminus = np.sqrt((xi - g)**2 + 0.25**2)
                expected = np.sort([-Eplus, -Eminus, Eminus, Eplus])
                assert_allclose(np.linalg.eigvalsh(bulk_bdg(kx, ky, alpha=alpha, Delta_t=0)),
                                expected, atol=1e-13)


class StripTests(unittest.TestCase):
    def test_hermiticity_particle_hole_time_reversal_and_antisymmetry(self):
        Ny = 4
        C = np.kron(SX, np.eye(2 * Ny))
        T = np.kron(np.eye(2 * Ny), 1j * SY)
        for kx in (0.0, 0.71, -1.2):
            with self.subTest(kx=kx):
                H, Hminus = strip_bdg(kx, Ny), strip_bdg(-kx, Ny)
                assert_allclose(H, H.conj().T, atol=1e-13)
                assert_allclose(C @ H.conj() @ C.conj().T, -Hminus, atol=1e-13)
                assert_allclose(T @ H.conj() @ T.conj().T, Hminus, atol=1e-13)
                _, Delta = build_blocks(kx, Ny)
                _, Delta_minus = build_blocks(-kx, Ny)
                assert_allclose(Delta, -Delta_minus.T, atol=1e-13)

    def test_open_chain_singlet_standing_waves(self):
        Ny, kx = 5, 0.71
        ky = np.pi * np.arange(1, Ny + 1) / (Ny + 1)
        xi = -2 * (np.cos(kx) + np.cos(ky)) + 2
        E = np.sqrt(xi**2 + 0.25**2)
        expected = np.sort(np.concatenate((-E, -E, E, E)))
        assert_allclose(np.linalg.eigvalsh(strip_bdg(kx, Ny, alpha=0, Delta_t=0)),
                        expected, atol=1e-13)

    def test_csv_retains_energies_outside_plot_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("make_figS1_edge_spectrum_nt.save_hd", side_effect=lambda fig, _: plt.close(fig)):
                make_strip_spectrum(Path(tmpdir), Ny=1, Nkx=3)
            data = pd.read_csv(Path(tmpdir) / "figS1_edge_spectrum_nt_exact_hd.csv")
        # For one y site at kx=-pi: xi=4, d=g=0, E=±sqrt(16+Delta_s^2).
        E = np.sqrt(16 + 0.25**2)
        assert_allclose(data.iloc[0, 1:].to_numpy(), [-E, -E, E, E], atol=1e-13)


if __name__ == "__main__":
    unittest.main()
