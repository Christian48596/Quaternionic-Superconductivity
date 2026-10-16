#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_chi_convergence_correlators_SM.py

Supplemental-material figure: chi-convergence of real-space correlators at fixed (U, L).

INPUT
-----
TeNPy CSV files named like:
  correlators_dmrg_quartet_L80_t1.0_tp0.6_U4.0_mu-1.2_Jp0.0_N160_chi800.csv

Each correlators_*.csv must contain columns:
  r, C2e(r), C4e(r)
(with an optional header line starting with "r," and/or comment lines starting with "#").

WHAT THIS PLOTS (3 panels)
--------------------------
(a) |C_{2e}(r)| for several chi values (log-log)
(b) |C_{4e}(r)| for several chi values (log-log)
(c) Relative deviations vs the largest chi (chi_ref = max chis):
      δ_i(chi; r) = |C_i^chi(r) - C_i^ref(r)| / |C_i^ref(r)|,   i = 2e, 4e
    computed only where |C_i^ref(r)| >= floor (to avoid noise blow-ups).
    (x is log, y is log)

STYLE / LAYOUT
--------------
- Panel letters (a)(b)(c) are NOT bold and are placed further left (outside axes).
- Limits are padded so markers are NOT clipped by the axes borders.
- Safe log-x padding avoids the "non-positive xlim on log axis" warning.

EXAMPLE
-------
python plot_chi_convergence_correlators_SM.py \
  --data_dir . --out_dir .. \
  --U 4.0 --L 80 --chis 800,1000,1200 \
  --rmax 40 --floor 1e-16 \
  --png --dpi 800
"""

from __future__ import annotations

import os
import re
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Small utilities
# ----------------------------
def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def pad_limits_logx(x: np.ndarray, frac: float = 0.08) -> Tuple[float, float]:
    """
    Safe padding for log-x axes: uses log10 padding and enforces xmin>0.
    Larger frac => more breathing room so endpoints/markers never touch borders.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size == 0:
        return 1.0, 10.0
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return max(1e-12, xmin * 0.85), xmax * 1.15
    lx0, lx1 = np.log10(xmin), np.log10(xmax)
    dl = (lx1 - lx0) * float(frac)
    lo = 10 ** (lx0 - dl)
    hi = 10 ** (lx1 + dl)
    lo = max(lo, 1e-12)
    return lo, hi


def set_ylim_from_data_log(ax, arrays: List[np.ndarray], pad_decades: float = 0.45) -> None:
    """
    Set log-y limits from arrays (positive finite values only),
    padded in decades so markers are not clipped.
    """
    vals = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a) & (a > 0.0)]
        if a.size:
            vals.append(a)
    if not vals:
        return
    v = np.concatenate(vals)
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo <= 0.0:
        lo = np.min(v[v > 0.0])
    lo2 = 10 ** (np.log10(lo) - float(pad_decades))
    hi2 = 10 ** (np.log10(hi) + float(pad_decades))
    ax.set_ylim(lo2, hi2)


def add_panel_letter(
    ax,
    letter: str,
    x: float = -0.22,
    y: float = 1.02,
    fontsize: float = 16.0,
) -> None:
    """
    Place (a)/(b)/(c) outside the axes, not bold.
    Tune x more negative to move further left.
    """
    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=fontsize,
        fontweight="normal",
        clip_on=False,
    )


# ----------------------------
# I/O
# ----------------------------
def read_correlators_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_list, c2_list, c4_list = [], [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("r,"):
                continue
            parts = s.split(",")
            if len(parts) < 3:
                continue
            r_list.append(int(parts[0]))
            c2_list.append(float(parts[1]))
            c4_list.append(float(parts[2]))

    r = np.array(r_list, dtype=int)
    c2 = np.array(c2_list, dtype=float)
    c4 = np.array(c4_list, dtype=float)

    idx = np.argsort(r)
    return r[idx], c2[idx], c4[idx]


def find_correlators_file(data_dir: str, U: float, L: int, chi: int, recursive: bool) -> str:
    pat = os.path.join(data_dir, "**", "correlators_*.csv") if recursive else os.path.join(data_dir, "correlators_*.csv")
    files = sorted(glob.glob(pat, recursive=recursive))

    def match(p: str) -> bool:
        b = os.path.basename(p)
        mL = re.search(r"_L([0-9]+)_", b)
        mU = re.search(r"_U([-0-9\.eE]+)_", b)
        mC = re.search(r"_chi([0-9]+)\.csv$", b)
        if not (mL and mU and mC):
            return False
        Lf = int(mL.group(1))
        Uf = float(mU.group(1))
        Cf = int(mC.group(1))
        if Lf != int(L):
            return False
        if Cf != int(chi):
            return False
        return abs(Uf - float(U)) <= 1e-12

    hits = [p for p in files if match(p)]
    if not hits:
        raise FileNotFoundError(f"No correlators file for U={U}, L={L}, chi={chi} in '{data_dir}' (recursive={recursive}).")
    return hits[-1]


# ----------------------------
# Main plotting
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot chi-convergence of correlators |C2e(r)|, |C4e(r)| and relative deviations vs chi_ref.",
    )

    ap.add_argument("--data_dir", type=str, default=".", help="Directory containing correlators_*.csv")
    ap.add_argument("--recursive", action="store_true", help="Search correlators_*.csv recursively under data_dir")
    ap.add_argument("--out_dir", type=str, default=".", help="Output directory")
    ap.add_argument("--out_prefix", type=str, default="dmrg_quartet", help="Tag for output filename")

    ap.add_argument("--U", type=float, required=True, help="Interaction U (must match filename _U..._)")
    ap.add_argument("--L", type=int, required=True, help="System size L (must match filename _L..._)")
    ap.add_argument("--chis", type=str, required=True, help="Comma-separated chis, e.g. 800,1000,1200")

    ap.add_argument("--rmax", type=int, default=None, help="Optional max r to plot (applied after reading)")
    ap.add_argument("--floor", type=float, default=1e-16,
                    help="Noise floor used ONLY in panel (c) masking: require |C_ref(r)| >= floor.")
    ap.add_argument("--eps", type=float, default=1e-300, help="Tiny epsilon to protect log plots")

    # layout / style
    ap.add_argument("--figsize", type=str, default="7.2,8.2", help="Figure size W,H in inches, e.g. 7.2,8.2")
    ap.add_argument("--hspace", type=float, default=0.18, help="Vertical spacing between panels")
    ap.add_argument("--labelsize", type=float, default=18.0)
    ap.add_argument("--ticksize", type=float, default=14.0)
    ap.add_argument("--legendsize", type=float, default=12.0)
    ap.add_argument("--linewidth", type=float, default=1.8)
    ap.add_argument("--markersize", type=float, default=4.5)

    # panel letters
    ap.add_argument("--letter_x", type=float, default=-0.22, help="Panel letter x in axes fraction (more negative = more left)")
    ap.add_argument("--letter_y", type=float, default=1.02, help="Panel letter y in axes fraction")
    ap.add_argument("--letter_size", type=float, default=16.0, help="Panel letter fontsize")

    # padding knobs to prevent marker clipping
    ap.add_argument("--xpad_frac", type=float, default=0.08, help="Extra log-x padding fraction (bigger => more space at ends)")
    ap.add_argument("--ypad_decades", type=float, default=0.45, help="Extra log-y padding in decades (bigger => less clipping)")

    # output
    ap.add_argument("--png", action="store_true", help="Also export PNG")
    ap.add_argument("--dpi", type=int, default=800, help="PNG DPI")
    ap.add_argument("--pad_inches", type=float, default=0.15, help="Extra padding for saved figure")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chis = parse_csv_ints(args.chis)
    if len(chis) < 2:
        raise ValueError("Need at least two chis to show convergence.")

    chi_ref = max(chis)

    # Read all chis
    series: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chi in chis:
        path = find_correlators_file(args.data_dir, U=args.U, L=args.L, chi=chi, recursive=args.recursive)
        r, c2, c4 = read_correlators_csv(path)
        if args.rmax is not None:
            m = (r <= int(args.rmax))
            r, c2, c4 = r[m], c2[m], c4[m]
        series[int(chi)] = (r, c2, c4)

    # Common r-grid (intersection) for panel (c)
    r_sets = [set(series[chi][0].astype(int).tolist()) for chi in chis]
    r_common = sorted(set.intersection(*r_sets))
    if not r_common:
        raise RuntimeError("No common r values across chis. (Panel c needs common r.)")
    r_grid = np.array(r_common, dtype=int)

    # Interpolate by direct indexing (r values are integers)
    def values_on_grid(chi: int) -> Tuple[np.ndarray, np.ndarray]:
        r, c2, c4 = series[chi]
        idx = {int(rv): i for i, rv in enumerate(r.astype(int))}
        c2g = np.array([c2[idx[int(rv)]] for rv in r_grid], dtype=float)
        c4g = np.array([c4[idx[int(rv)]] for rv in r_grid], dtype=float)
        return c2g, c4g

    c2_ref, c4_ref = values_on_grid(chi_ref)

    # Relative deviations (masked where ref is below floor)
    mask2 = np.abs(c2_ref) >= float(args.floor)
    mask4 = np.abs(c4_ref) >= float(args.floor)

    deltas_2e: Dict[int, np.ndarray] = {}
    deltas_4e: Dict[int, np.ndarray] = {}

    for chi in chis:
        c2g, c4g = values_on_grid(chi)
        d2 = np.full_like(c2g, np.nan, dtype=float)
        d4 = np.full_like(c4g, np.nan, dtype=float)
        d2[mask2] = np.abs(c2g[mask2] - c2_ref[mask2]) / np.abs(c2_ref[mask2])
        d4[mask4] = np.abs(c4g[mask4] - c4_ref[mask4]) / np.abs(c4_ref[mask4])
        deltas_2e[chi] = d2
        deltas_4e[chi] = d4

    # ---- figure ----
    W, H = [float(x.strip()) for x in args.figsize.split(",")]
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(3, 1, hspace=float(args.hspace))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    # (a) |C2e|
    for chi in chis:
        r, c2, _ = series[chi]
        ax1.loglog(
            r.astype(float),
            np.abs(c2) + float(args.eps),
            marker="o",
            markersize=args.markersize,
            linewidth=args.linewidth,
            label=rf"$\chi={chi}$",
        )
    ax1.set_ylabel(r"$|C_{2e}(r)|$", fontsize=args.labelsize)
    ax1.tick_params(axis="both", labelsize=args.ticksize)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=args.legendsize, frameon=True, loc="best")
    add_panel_letter(ax1, "a", x=args.letter_x, y=args.letter_y, fontsize=args.letter_size)

    r_all_a = np.concatenate([series[chi][0].astype(float) for chi in chis])
    ax1.set_xlim(*pad_limits_logx(r_all_a, frac=float(args.xpad_frac)))
    set_ylim_from_data_log(ax1, [np.abs(series[chi][1]) + float(args.eps) for chi in chis],
                           pad_decades=float(args.ypad_decades))

    # (b) |C4e|
    for chi in chis:
        r, _, c4 = series[chi]
        ax2.loglog(
            r.astype(float),
            np.abs(c4) + float(args.eps),
            marker="s",
            markersize=args.markersize,
            linewidth=args.linewidth,
            linestyle="--",
            label=rf"$\chi={chi}$",
        )
    ax2.set_ylabel(r"$|C_{4e}(r)|$", fontsize=args.labelsize)
    ax2.tick_params(axis="both", labelsize=args.ticksize)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=args.legendsize, frameon=True, loc="best")
    add_panel_letter(ax2, "b", x=args.letter_x, y=args.letter_y, fontsize=args.letter_size)

    r_all_b = np.concatenate([series[chi][0].astype(float) for chi in chis])
    ax2.set_xlim(*pad_limits_logx(r_all_b, frac=float(args.xpad_frac)))
    set_ylim_from_data_log(ax2, [np.abs(series[chi][2]) + float(args.eps) for chi in chis],
                           pad_decades=float(args.ypad_decades))

    # (c) relative deviations vs chi_ref
    for chi in chis:
        if chi == chi_ref:
            continue
        d2 = deltas_2e[chi]
        d4 = deltas_4e[chi]

        m2 = np.isfinite(d2) & (d2 > 0)
        m4 = np.isfinite(d4) & (d4 > 0)

        if np.any(m2):
            ax3.semilogy(
                r_grid[m2].astype(float),
                d2[m2],
                marker="o",
                markersize=args.markersize,
                linewidth=args.linewidth,
                label=rf"$\delta_{{2e}}(\chi={chi})$",
            )
        if np.any(m4):
            ax3.semilogy(
                r_grid[m4].astype(float),
                d4[m4],
                marker="s",
                markersize=args.markersize,
                linewidth=args.linewidth,
                linestyle="--",
                label=rf"$\delta_{{4e}}(\chi={chi})$",
            )

    ax3.set_xlabel(r"distance $r$ (rungs, $a=1$)", fontsize=args.labelsize)
    ax3.set_ylabel(r"$\delta_i(r;\chi_{\max})$", fontsize=args.labelsize)
    ax3.tick_params(axis="both", labelsize=args.ticksize)
    ax3.grid(True, which="both", alpha=0.25)
    ax3.legend(fontsize=args.legendsize, frameon=True, loc="best")
    add_panel_letter(ax3, "c", x=args.letter_x, y=args.letter_y, fontsize=args.letter_size)

    ax3.set_xscale("log")
    ax3.set_xlim(*pad_limits_logx(r_grid.astype(float), frac=float(args.xpad_frac)))

    ylist_c = []
    for chi in chis:
        if chi == chi_ref:
            continue
        ylist_c.append(deltas_2e[chi])
        ylist_c.append(deltas_4e[chi])
    set_ylim_from_data_log(ax3, ylist_c, pad_decades=float(args.ypad_decades))

    # Save
    out_pdf = os.path.join(args.out_dir, f"fig_chi_convergence_corr_U{args.U}_L{args.L}_{args.out_prefix}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=float(args.pad_inches))
    if args.png:
        out_png = os.path.splitext(out_pdf)[0] + ".png"
        fig.savefig(out_png, bbox_inches="tight", pad_inches=float(args.pad_inches), dpi=int(args.dpi))
    plt.close(fig)

    print(f"[OK] wrote {out_pdf}")
    print(f"[INFO] chis: {chis} (reference chi_max={chi_ref})")
    print(f"[INFO] delta panel floor: |C_ref| >= {args.floor:g}")


if __name__ == "__main__":
    main()
