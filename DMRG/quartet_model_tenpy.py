#!/usr/bin/env python3
"""
quartet_model_tenpy.py
(unchanged physics; only robustness defaults for canonical runs)
"""

from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.networks.site import FermionSite


def u_index(orb, spin):
    if orb == 1 and spin == "up":
        return 0
    if orb == 1 and spin == "down":
        return 1
    if orb == 2 and spin == "up":
        return 2
    if orb == 2 and spin == "down":
        return 3
    raise ValueError(f"bad (orb,spin)=({orb},{spin})")


class TwoOrbitalQuartetChain(CouplingModel, MPOModel):
    def __init__(self, model_params):
        L = int(model_params.get("L", 16))
        t = float(model_params.get("t", 1.0))
        t_perp = float(model_params.get("t_perp", 0.6))
        U = float(model_params.get("U", 0.0))
        mu = float(model_params.get("mu", 0.0))

        bc = model_params.get("bc", "open")
        bc_MPS = model_params.get("bc_MPS", "finite")

        # IMPORTANT: for canonical fixed-N runs use conserve="N"
        conserve = model_params.get("conserve", "N")

        # Some TeNPy versions want bc as list per dimension
        if isinstance(bc, str):
            bc = [bc]

        self.L_rung = L
        self.t = t
        self.t_perp = t_perp
        self.U = U
        self.mu = mu

        # 4 Site objects in unit cell
        unit_cell = [FermionSite(conserve=conserve) for _ in range(4)]

        lat = Lattice([L], unit_cell, bc=bc, bc_MPS=bc_MPS)
        CouplingModel.__init__(self, lat)

        # onsite: -mu * N
        for u in range(4):
            self.add_onsite(-mu, u, "N")

        # chain hopping: -t between same u in neighboring unit cells
        for u in range(4):
            self.add_coupling(-t, u, "Cd", u, "C", dx=[1], plus_hc=True)

        # rung hopping: -t_perp between orbitals at same rung (spin-conserving)
        self.add_coupling(-t_perp, 0, "Cd", 2, "C", dx=[0], plus_hc=True)  # up
        self.add_coupling(-t_perp, 1, "Cd", 3, "C", dx=[0], plus_hc=True)  # down

        # -U * n1*n2 (expanded)
        if abs(U) > 0.0:
            for a in (0, 1):
                for b in (2, 3):
                    self.add_coupling(-U, a, "N", b, "N", dx=[0])

        H_MPO = self.calc_H_MPO()
        MPOModel.__init__(self, lat, H_MPO)
