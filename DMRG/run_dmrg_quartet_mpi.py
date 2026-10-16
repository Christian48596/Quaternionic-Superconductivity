#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dmrg_quartet_mpi.py

MPI parameter-scan driver for TeNPy DMRG on the two-orbital quartet chain.
For each parameter point, the script runs a ground-state DMRG calculation and
exports real-space charge-2e and charge-4e correlators plus summary diagnostics.

--------------------------------------------------------------------------------
What this script does
--------------------------------------------------------------------------------
1) Builds the model:
     TwoOrbitalQuartetChain (TeNPy model defined elsewhere in this repository)

2) Runs DMRG (ground state) with optional particle-number conservation:
     - typical production setting: conserve = "N" (U(1) particle number)
     - canonical half-filling is achieved by fixing the target particle number

3) Measures and saves observables:
     - real-space correlators:
         C_{2e}(r), C_{4e}(r)  vs rung separation r
     - structure/summary quantities at k=0 (or integrated quantities depending on implementation)
     - full run logs for reproducibility

4) Parallelization model (MPI):
     - MPI is used to distribute independent (L, U, chi) jobs across ranks.
     - Each rank runs its assigned jobs serially and writes unique output files.
     - There is no intra-run parallelization inside TeNPy DMRG; MPI is across the scan grid.

--------------------------------------------------------------------------------
Important compatibility note (TeNPy versions)
--------------------------------------------------------------------------------
Some TeNPy releases do NOT provide `psi.normalize()` for the MPS object.
To avoid crashes and to ensure all reported observables correspond to a normalized
state, this script uses a robust normalization strategy:

  - try to canonicalize the MPS if available (psi.canonical_form())
  - compute the norm:  norm2 = <psi|psi>
  - renormalize measured observables by dividing by norm2

This makes correlators and derived quantities consistent with a normalized |psi⟩
even when `psi.normalize()` is missing.

--------------------------------------------------------------------------------
Parameter scans supported
--------------------------------------------------------------------------------
The scan grid can include:
  - system sizes L (rungs):            --Ls
  - interactions U:                    --Us
  - bond dimensions (chi scan):        --chis

If --chis is provided, the full job grid becomes:
    (#Ls) × (#Us) × (#chis)

This is useful for convergence studies (e.g., referee requests).

--------------------------------------------------------------------------------
Outputs (per parameter point)
--------------------------------------------------------------------------------
For each job (unique L, U, chi), the script writes:

  correlators_<base>.csv
      Real-space correlators vs distance r:
        r, C2e(r), C4e(r), ... (depending on implementation)

  structure_<base>.csv
      Compact run summary:
        L, N_target, psi_norm2, (optional) S2e0, S4e0, ...

  log_<base>.txt
      Full diagnostics and DMRG output (energies, truncation errors, sweeps, etc.)

All filenames include enough metadata (L, U, chi, model parameters) to avoid clashes
when running under MPI.

--------------------------------------------------------------------------------
Performance / MPI tips
--------------------------------------------------------------------------------
MPI distributes jobs across ranks, but BLAS libraries (MKL/OpenBLAS/Accelerate)
may still spawn multiple threads per rank, causing oversubscription.

For best performance, set BLAS threads to 1 per rank:
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1

--------------------------------------------------------------------------------
Examples
--------------------------------------------------------------------------------
1) Basic (L,U) scan at fixed chi:
  mpirun -np 8 python run_dmrg_quartet_mpi.py \
    --Ls 16,24,32,40 \
    --Us 2.5,4,6,8,9 \
    --t 1.0 --t_perp 0.6 --mu -1.2 \
    --conserve N --chi_max 800 \
    --out_dir results --out_prefix dmrg_quartet

2) Bond-dimension convergence scan at fixed (L,U) and fixed N (half filling):
  mpirun -np 8 python run_dmrg_quartet_mpi.py \
    --Ls 80 --Us 4.0 \
    --chis 800,1000,1200,1600 \
    --t 1.0 --t_perp 0.6 --mu -1.2 \
    --conserve N --N_target 160 \
    --out_dir results --out_prefix dmrg_quartet

Notes:
- L is the number of rungs (unit cells).
- With conserve="N", use --N_target to fix the total particle number (e.g., N=2L at half filling).
- Output files are written into --out_dir with prefix --out_prefix.
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
import warnings
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# -----------------------------------------------------------------------------
# (Optional) Set these BEFORE numpy import if you want the script to force 1 thread
# per MPI rank by default. Usually you export them in the shell instead.
# -----------------------------------------------------------------------------
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

from quartet_model_tenpy import TwoOrbitalQuartetChain, u_index


# -----------------------------------------------------------------------------
# Reduce TeNPy logging noise (keeps YOUR prints/logs)
# -----------------------------------------------------------------------------
logging.getLogger("tenpy").setLevel(logging.ERROR)

# Suppress specific TeNPy warnings you reported (safe to ignore for production runs)
warnings.filterwarnings(
    "ignore",
    message=r".*SVD with lapack_driver 'gesdd' failed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*final DMRG state not in canonical form.*",
    category=UserWarning,
)


# -----------------------------------------------------------------------------
# MPI (optional): runs serial if mpi4py isn't available
# -----------------------------------------------------------------------------
def mpi_setup():
    """Return (comm, rank, size). If mpi4py missing, return dummy serial comm."""
    try:
        from mpi4py import MPI  # type: ignore
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        class DummyComm:
            def bcast(self, x, root=0): return x
            def Barrier(self): return None
            def gather(self, x, root=0): return [x]
        return DummyComm(), 0, 1


COMM, RANK, SIZE = mpi_setup()


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------
def parse_csv_floats(s: Optional[str]) -> List[float]:
    if s is None or s.strip() == "":
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_csv_ints(s: Optional[str]) -> List[int]:
    if s is None or s.strip() == "":
        return []
    return [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]


# -----------------------------------------------------------------------------
# Index mapping: (rung i, orb, spin) -> flat MPS site index p
# -----------------------------------------------------------------------------
def flat_index(i: int, orb: int, spin: str, L: int) -> int:
    u = u_index(orb, spin)
    return 4 * i + u


# -----------------------------------------------------------------------------
# Build initial product state with EXACT particle number
# -----------------------------------------------------------------------------
def build_initial_state_fixed_N(model: TwoOrbitalQuartetChain,
                                N_target: int,
                                pattern: str = "uniform2perrung") -> List[str]:
    """
    Build a product state with EXACT N_target occupied spinless-fermion modes.
    Total modes = 4L (spinless FermionSite per mode).

    Recommended for half filling: N_target = 2L.

    pattern:
      - uniform2perrung: fill u=0 and u=2 on each rung (2 per rung)
      - center_block: contiguous block of N_target filled sites centered
    """
    L_tot = model.lat.N_sites
    if not (0 <= N_target <= L_tot):
        raise ValueError(f"N_target must be in [0, {L_tot}], got {N_target}")

    occ = np.zeros(L_tot, dtype=int)

    if pattern == "uniform2perrung":
        L = model.L_rung
        for i in range(L):
            for u in (0, 2):
                occ[4 * i + u] = 1
        current = int(occ.sum())

        # deterministic adjust if N_target != 2L
        if current > N_target:
            ones = np.where(occ == 1)[0]
            remove = ones[::-1][: (current - N_target)]
            occ[remove] = 0
        elif current < N_target:
            zeros = np.where(occ == 0)[0]
            add = zeros[: (N_target - current)]
            occ[add] = 1

    elif pattern == "center_block":
        start = (L_tot - N_target) // 2
        occ[start:start + N_target] = 1
    else:
        raise ValueError(f"Unknown pattern '{pattern}'")

    return ["full" if x == 1 else "empty" for x in occ.tolist()]


# -----------------------------------------------------------------------------
# Fermionic operator products with explicit JW strings
# -----------------------------------------------------------------------------
def apply_fermion_product_with_JW(psi: MPS, ops: List[str], sites: List[int]) -> MPS:
    """
    Apply product of fermionic operators (C/Cd) with explicit JW strings.

    Even-number assumption holds for:
      Delta: 2 ops
      Q:     4 ops

    JW is inserted STRICTLY BETWEEN paired operators:
      after op #0 until before op #1,
      after op #2 until before op #3, etc.
    i.e. range(s_m + 1, s_next)
    """
    if len(ops) != len(sites):
        raise ValueError("ops and sites must have the same length.")
    if len(ops) % 2 != 0:
        raise ValueError("Need an even number of fermion ops for this JW helper.")
    for o in ops:
        if o not in ("C", "Cd"):
            raise ValueError(f"Unsupported fermion op '{o}' (expected 'C' or 'Cd').")

    items = sorted(zip(sites, ops), key=lambda x: x[0])
    sites_s = [s for s, _ in items]
    ops_s = [o for _, o in items]

    psi_tmp = psi.copy()

    for m in range(len(ops_s) - 1):
        s_m = sites_s[m]
        op_m = ops_s[m]
        psi_tmp.apply_local_op(s_m, psi_tmp.sites[s_m].get_op(op_m))

        if (m % 2) == 0:
            s_next = sites_s[m + 1]
            for k in range(s_m + 1, s_next):
                psi_tmp.apply_local_op(k, psi_tmp.sites[k].get_op("JW"))

    s_last = sites_s[-1]
    op_last = ops_s[-1]
    psi_tmp.apply_local_op(s_last, psi_tmp.sites[s_last].get_op(op_last))

    return psi_tmp


# -----------------------------------------------------------------------------
# Correlators + structure factors (renormalized by psi norm)
# -----------------------------------------------------------------------------
def correlators_dmrg(model: TwoOrbitalQuartetChain,
                     psi: MPS,
                     max_r: Optional[int] = None,
                     i0: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """
    Compute:
      C2e(r) = <Δ_i0^† Δ_{i0+r}>
      C4e(r) = <Q_i0^† Q_{i0+r}>

    and
      S2e0 = 1/L * || Δ_tot |psi> ||^2
      S4e0 = 1/L * || Q_tot |psi> ||^2

    Returns also psi_norm2 = <psi|psi> used for renormalization.

    IMPORTANT:
    Our overlaps are computed on (possibly slightly unnormalized) psi.
    For a non-unit-norm psi with norm2=n, all expectation-like overlaps scale by n.
    Therefore we divide correlators and structure factors by psi_norm2.
    """
    L = model.L_rung
    if max_r is None:
        max_r = L // 2
    if i0 is None:
        i0 = L // 2

    # robust norm2 (works across TeNPy versions)
    psi_norm2 = float(np.real(psi.overlap(psi)))
    inv_norm2 = 1.0 / psi_norm2 if psi_norm2 != 0.0 else 1.0

    def rung_indices(i: int):
        p1u = flat_index(i, 1, "up",   L)
        p1d = flat_index(i, 1, "down", L)
        p2u = flat_index(i, 2, "up",   L)
        p2d = flat_index(i, 2, "down", L)
        return p1u, p1d, p2u, p2d

    Delta_ops: Dict[int, List[Tuple[List[str], List[int], float]]] = {}
    Q_ops: Dict[int, Tuple[List[str], List[int]]] = {}

    for i in range(L):
        p1u, p1d, p2u, p2d = rung_indices(i)
        Delta_ops[i] = [
            (["C", "C"], [p1u, p2d], +1.0),
            (["C", "C"], [p1d, p2u], -1.0),
        ]
        Q_ops[i] = (["C", "C", "C", "C"], [p1u, p1d, p2u, p2d])

    def apply_Delta(i: int, psi_ref: MPS) -> MPS:
        (ops1, sites1, _), (ops2, sites2, c2) = Delta_ops[i]
        psi_a = apply_fermion_product_with_JW(psi_ref, ops1, sites1)
        psi_b = apply_fermion_product_with_JW(psi_ref, ops2, sites2)
        return psi_a.add(psi_b, alpha=1.0, beta=c2)  # c2 = -1

    def apply_Q(i: int, psi_ref: MPS) -> MPS:
        ops, sites = Q_ops[i]
        return apply_fermion_product_with_JW(psi_ref, ops, sites)

    # ---- correlators ----
    r_list, C2_list, C4_list = [], [], []

    psi_D_i0 = apply_Delta(i0, psi)
    psi_Q_i0 = apply_Q(i0, psi)

    for r in range(1, max_r + 1):
        j = i0 + r
        if j >= L:
            break

        psi_D_j = apply_Delta(j, psi)
        C2 = psi_D_i0.overlap(psi_D_j)

        psi_Q_j = apply_Q(j, psi)
        C4 = psi_Q_i0.overlap(psi_Q_j)

        r_list.append(r)
        C2_list.append(float(np.real(C2)) * inv_norm2)
        C4_list.append(float(np.real(C4)) * inv_norm2)

    r_arr = np.array(r_list, dtype=int)
    C2e_arr = np.array(C2_list, dtype=float)
    C4e_arr = np.array(C4_list, dtype=float)

    # ---- structure factors at q=0 ----
    psi_D_tot = None
    psi_Q_tot = None
    for i in range(L):
        psi_D_i = apply_Delta(i, psi)
        psi_Q_i = apply_Q(i, psi)
        if psi_D_tot is None:
            psi_D_tot = psi_D_i
            psi_Q_tot = psi_Q_i
        else:
            psi_D_tot = psi_D_tot.add(psi_D_i, alpha=1.0, beta=1.0)
            psi_Q_tot = psi_Q_tot.add(psi_Q_i, alpha=1.0, beta=1.0)

    S2e0_raw = float(np.real(psi_D_tot.overlap(psi_D_tot))) / L
    S4e0_raw = float(np.real(psi_Q_tot.overlap(psi_Q_tot))) / L

    S2e0 = S2e0_raw * inv_norm2
    S4e0 = S4e0_raw * inv_norm2

    return r_arr, C2e_arr, C4e_arr, S2e0, S4e0, psi_norm2


# -----------------------------------------------------------------------------
# DMRG diagnostics printing (works across TenPy versions)
# -----------------------------------------------------------------------------
def print_dmrg_diagnostics(info: Dict[str, Any], stream):
    print("[INFO] DMRG keys:", list(info.keys()), file=stream)
    if "E" in info:
        print("[INFO] E =", info["E"], file=stream)

    ss = info.get("sweep_statistics", None)
    if isinstance(ss, dict):
        print("[INFO] sweep_statistics keys:", list(ss.keys()), file=stream)
        for k in ["E", "Delta_E", "max_trunc_err", "trunc_err", "E_trunc"]:
            if k in ss:
                v = ss[k]
                try:
                    print(f"[INFO] last {k} =", v[-1], file=stream)
                except Exception:
                    print(f"[INFO] {k} =", v, file=stream)

    bs = info.get("bond_statistics", None)
    if isinstance(bs, dict):
        print("[INFO] bond_statistics keys:", list(bs.keys()), file=stream)


# -----------------------------------------------------------------------------
# Job definition
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    L: int
    t: float
    t_perp: float
    U: float
    mu: float
    J_pair: float
    conserve: str
    N_target: int
    chi_max: int
    svd_min: float
    mixer: bool
    max_E_err: float
    combine: bool
    i0: Optional[int]
    max_r: Optional[int]
    init_pattern: str


def job_base_name(job: Job, out_prefix: str) -> str:
    return (
        f"{out_prefix}_L{job.L}_t{job.t}_tp{job.t_perp}"
        f"_U{job.U}_mu{job.mu}_Jp{job.J_pair}_N{job.N_target}_chi{job.chi_max}"
    )


# -----------------------------------------------------------------------------
# Run one job (DMRG + outputs)
# -----------------------------------------------------------------------------
def run_one_job(job: Job, out_prefix: str, out_dir: str, write_log: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    base = job_base_name(job, out_prefix)

    log_path = os.path.join(out_dir, f"log_{base}.txt")
    corr_path = os.path.join(out_dir, f"correlators_{base}.csv")
    struct_path = os.path.join(out_dir, f"structure_{base}.csv")

    stream = open(log_path, "w") if write_log else sys.stdout

    try:
        print("=" * 80, file=stream)
        print(f"[RANK {RANK}] Running job: {base}", file=stream)
        print(job, file=stream)
        print("=" * 80, file=stream)
        stream.flush()

        # ---- build model ----
        model_params = dict(
            L=job.L,
            t=job.t,
            t_perp=job.t_perp,
            U=job.U,
            mu=job.mu,
            J_pair=job.J_pair,
            bc_MPS="finite",
            conserve=job.conserve,
        )
        model = TwoOrbitalQuartetChain(model_params)

        # ---- initial MPS (exact N) ----
        product_state = build_initial_state_fixed_N(model, N_target=job.N_target, pattern=job.init_pattern)
        psi = MPS.from_product_state(
            model.lat.mps_sites(),
            product_state,
            bc=model.lat.bc_MPS,
            unit_cell_width=model.lat.mps_unit_cell_width,
        )

        # ---- DMRG ----
        dmrg_params = {
            "mixer": job.mixer,
            "max_E_err": job.max_E_err,
            "trunc_params": {"chi_max": job.chi_max, "svd_min": job.svd_min},
            "combine": job.combine,
        }

        info = dmrg.run(psi, model, dmrg_params)
        E0 = float(info["E"])
        print(f"[INFO] DMRG finished: E0 = {E0:.12f}", file=stream)
        print("[INFO] final bond dimensions:", psi.chi, file=stream)
        print_dmrg_diagnostics(info, stream)

        # ---- canonicalize if available ----
        try:
            psi.canonical_form()
            print("[INFO] canonical_form() applied.", file=stream)
        except Exception as e:
            print(f"[WARN] psi.canonical_form() not applied ({e}). Continuing.", file=stream)

        # ---- density diagnostic ----
        n_site = np.array([psi.expectation_value("N", i) for i in range(model.lat.N_sites)], dtype=float)
        Ntot = float(n_site.sum())
        print(f"[INFO] <N_total> = {Ntot:.6f}   per rung = {Ntot/job.L:.6f}   (target {job.N_target})", file=stream)

        # ---- correlators + structure (renormalized) ----
        r, C2e, C4e, S2e0, S4e0, psi_norm2 = correlators_dmrg(model, psi, max_r=job.max_r, i0=job.i0)
        print(f"[INFO] <psi|psi> = {psi_norm2:.12e}", file=stream)
        print(f"[INFO] S2e0 = {S2e0:.6e}   S4e0 = {S4e0:.6e}", file=stream)

        # ---- write correlators ----
        with open(corr_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# rank", RANK])
            w.writerow(["# L", job.L])
            w.writerow(["# E0", E0])
            w.writerow(["# N_target", job.N_target])
            w.writerow(["# psi_norm2", psi_norm2])
            w.writerow(["r", "C2e", "C4e"])
            for rr, c2, c4 in zip(r, C2e, C4e):
                w.writerow([int(rr), float(c2), float(c4)])

        # ---- write structure factors ----
        with open(struct_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["L", "N_target", "psi_norm2", "S2e0", "S4e0"])
            w.writerow([job.L, job.N_target, psi_norm2, float(S2e0), float(S4e0)])

        print(f"[INFO] Wrote '{corr_path}'", file=stream)
        print(f"[INFO] Wrote '{struct_path}'", file=stream)
        stream.flush()

        return dict(ok=True, base=base, E0=E0, Ntot=Ntot, S2e0=S2e0, S4e0=S4e0,
                    corr_path=corr_path, struct_path=struct_path, log_path=log_path)

    except Exception as e:
        print("[ERROR] Exception while running job:", file=stream)
        print(repr(e), file=stream)
        stream.flush()
        return dict(ok=False, base=base, error=str(e), log_path=log_path)

    finally:
        if write_log:
            stream.close()


# -----------------------------------------------------------------------------
# Build job list (Ls x Us x chis)
# -----------------------------------------------------------------------------
def build_jobs(args) -> List[Job]:
    L_list = parse_csv_ints(args.Ls) if args.Ls else [args.L]
    U_list = parse_csv_floats(args.Us) if args.Us else [args.U]

    # NEW: chi scan
    chi_list = parse_csv_ints(args.chis) if args.chis else [args.chi_max]

    jobs: List[Job] = []
    for L in L_list:
        N_target = args.N_target if args.N_target is not None else 2 * L  # half filling
        for U in U_list:
            for chi in chi_list:
                jobs.append(Job(
                    L=L,
                    t=args.t,
                    t_perp=args.t_perp,
                    U=U,
                    mu=args.mu,
                    J_pair=args.J_pair,
                    conserve=args.conserve,
                    N_target=N_target,
                    chi_max=chi,
                    svd_min=args.svd_min,
                    mixer=args.mixer,
                    max_E_err=args.max_E_err,
                    combine=args.combine,
                    i0=args.i0,
                    max_r=args.max_r,
                    init_pattern=args.init_pattern,
                ))
    return jobs


def distribute_jobs(jobs: List[Job], rank: int, size: int) -> List[Job]:
    return [job for i, job in enumerate(jobs) if (i % size) == rank]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "TeNPy DMRG for TwoOrbitalQuartetChain with canonical half-filling (fixed N) "
            "and charge-2e/4e correlators. MPI-parallel scan over L, U, and optionally chi."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # single-point defaults
    parser.add_argument("--L", type=int, default=32, help="Number of rungs (unit cells).")
    parser.add_argument("--t", type=float, default=1.0, help="Leg hopping t.")
    parser.add_argument("--t_perp", type=float, default=0.6, help="Rung hopping t_perp.")
    parser.add_argument("--U", type=float, default=2.5, help="Attraction strength (Hamiltonian uses -U n1 n2).")
    parser.add_argument("--mu", type=float, default=-1.2, help="Chemical potential.")
    parser.add_argument("--J_pair", type=float, default=0.0, help="Unused (kept for naming compatibility).")

    # scan controls
    parser.add_argument("--Ls", type=str, default=None, help="Comma-separated L list, e.g. '16,24,32,40'.")
    parser.add_argument("--Us", type=str, default=None, help="Comma-separated U list, e.g. '2.5,4,6,8,9'.")

    # NEW: chi scan list
    parser.add_argument("--chis", type=str, default=None,
                        help="Comma-separated chi_max list, e.g. '800,1000,1200,1600'. If omitted uses --chi_max.")

    # canonical controls
    parser.add_argument("--conserve", type=str, default="N",
                        help="Use 'N' to conserve particle number (canonical).")
    parser.add_argument("--N_target", type=int, default=None,
                        help="Total particle number. Default is 2*L (half filling).")
    parser.add_argument("--init_pattern", type=str, default="uniform2perrung",
                        choices=["uniform2perrung", "center_block"],
                        help="Initial product-state pattern at fixed N.")

    # dmrg numerics
    parser.add_argument("--chi_max", type=int, default=400, help="Max bond dimension (used if --chis not given).")
    parser.add_argument("--svd_min", type=float, default=1e-10, help="SVD cutoff in truncation.")
    parser.add_argument("--max_E_err", type=float, default=1e-8, help="DMRG energy convergence target.")
    parser.add_argument("--mixer", action="store_true", default=True, help="Enable DMRG mixer (default on).")
    parser.add_argument("--no_mixer", action="store_false", dest="mixer", help="Disable DMRG mixer.")
    parser.add_argument("--combine", action="store_true", default=True, help="Use combine=True (default on).")
    parser.add_argument("--no_combine", action="store_false", dest="combine", help="Disable combine.")

    # correlator options
    parser.add_argument("--i0", type=int, default=None, help="Rung index i0 (default L//2).")
    parser.add_argument("--max_r", type=int, default=None, help="Max distance r (default L//2).")

    # output
    parser.add_argument("--out_prefix", type=str, default="dmrg_quartet", help="Prefix for output filenames.")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory for outputs.")
    parser.add_argument("--no_log", action="store_true", help="Disable per-job log files.")

    args = parser.parse_args()

    # rank 0 builds jobs
    if RANK == 0:
        jobs = build_jobs(args)
        if len(jobs) == 0:
            raise RuntimeError("No jobs to run (empty job list).")
        print(f"[MPI] size={SIZE} ranks. Total jobs={len(jobs)}")

        # show scan axes
        L_desc = args.Ls if args.Ls else str(args.L)
        U_desc = args.Us if args.Us else str(args.U)
        chi_desc = args.chis if args.chis else str(args.chi_max)
        print(f"[MPI] Scan axes: Ls={L_desc}  Us={U_desc}  chis={chi_desc}")

    else:
        jobs = None

    jobs = COMM.bcast(jobs, root=0)

    my_jobs = distribute_jobs(jobs, RANK, SIZE)
    COMM.Barrier()
    print(f"[MPI] Rank {RANK}/{SIZE}: {len(my_jobs)} jobs", flush=True)
    COMM.Barrier()

    results = []
    for job in my_jobs:
        res = run_one_job(job, out_prefix=args.out_prefix, out_dir=args.out_dir, write_log=(not args.no_log))
        results.append(res)

    all_results = COMM.gather(results, root=0)

    if RANK == 0:
        flat = [x for sub in all_results for x in sub]
        ok = [r for r in flat if r.get("ok", False)]
        bad = [r for r in flat if not r.get("ok", False)]
        print("=" * 80)
        print(f"[MPI] Finished. OK={len(ok)}  FAIL={len(bad)}")
        if ok:
            print("[MPI] Summary (first 10 OK jobs):")
            for r in ok[:10]:
                print(f"  {r['base']}  E0={r['E0']:.10f}  S2e0={r['S2e0']:.3e}  S4e0={r['S4e0']:.3e}")
        if bad:
            print("[MPI] Failures:")
            for r in bad:
                print(f"  {r['base']}  error={r.get('error','?')}  log={r.get('log_path','?')}")
        print("=" * 80)


if __name__ == "__main__":
    main()