#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOW TO RUN
==========

This script reads a correlators CSV produced by the DMRG driver and generates a two-panel
diagnostic plot:
  (top)   local exponents eta_{2e}(r), eta_{4e}(r) from sliding log–log fits
  (bottom) tail ratio R_tail(L; r0) comparing long-distance weight of 4e vs 2e correlators

Typical input file name (example):
  correlators_dmrg_quartet_L80_t1.0_tp0.6_U4.0_mu-1.2_Jp0.0_N160_chi800.csv

In all plots:
  r  = rung separation (unit cells), lattice spacing a=1
  r0 = tail cutoff in the same units (rungs), a=1

USAGE (example, multiline)
--------------------------
IMPORTANT: the backslash "\" must be the LAST character on the line (no trailing spaces).

python plot_eta_Rtail_2panel.py \
  --corr_csv correlators_dmrg_quartet_L80_t1.0_tp0.6_U4.0_mu-1.2_Jp0.0_N160_chi800.csv \
  --out_dir . \
  --out_prefix fig_eta_Rtail_2panel_L80_U4.0_chi800 \
  --fit_window 7 \
  --eta_floor 1e-14 \
  --r_min 2 \
  --tail_floor 1e-18 \
  --r0_min 1 \
  --png \
  --dpi 800

USAGE (example, one line)
-------------------------
python plot_eta_Rtail_2panel.py --corr_csv correlators_dmrg_quartet_L80_t1.0_tp0.6_U4.0_mu-1.2_Jp0.0_N160_chi800.csv --out_dir . --out_prefix fig_eta_Rtail_2panel_L80_U4.0_chi800 --fit_window 7 --eta_floor 1e-14 --r_min 2 --tail_floor 1e-18 --r0_min 1 --png --dpi 800

OUTPUTS
-------
The script writes the figure(s) to:
  <out_dir>/<out_prefix>.png   (if --png is used)
  <out_dir>/<out_prefix>.pdf   (if a --pdf flag exists in this version / or if PDF is default)

If you do not see a PDF, check the script's argparse options:
  python plot_eta_Rtail_2panel.py --help

KEY PARAMETERS (what they do)
-----------------------------
--fit_window (int):
  Number of consecutive r-points used in each log–log least-squares fit. Larger -> smoother eta(r),
  but less local and may terminate earlier at large r.

--eta_floor (float):
  Threshold for "trusted window" rule. eta_i(r) is computed ONLY if *all* points in the fit window
  satisfy |C_i(r)| >= eta_floor. If eta(r) stops early, lower eta_floor and/or reduce fit_window.

--r_min (int):
  Minimum rung separation for the eta(r) analysis. Helps exclude short-distance nonuniversal regime.

--r0_min (int):
  Smallest tail cutoff r0 used in R_tail(L; r0). The script typically scans r0 from r0_min up to
  some maximum allowed by the data length.

--tail_floor (float):
  Floor used to avoid ratios dominated by numerical noise in very small correlator tails.

TROUBLESHOOTING
---------------
1) eta curves end too early:
   - lower --eta_floor (e.g. 1e-16)
   - reduce --fit_window (e.g. 5)
   - ensure the correlators CSV extends to large enough r

2) R_tail looks noisy:
   - raise --tail_floor slightly
   - restrict the plotted r0 range (if the script supports it)
   - verify correlators are properly normalized (psi norm handled in the DMRG driver)

3) Missing columns / wrong CSV format:
   - confirm the correlators CSV contains r and both 2e/4e correlator columns expected by this script
   - print the CSV header or open it to verify naming conventions

"""

from __future__ import annotations

import os
import re
import argparse
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# IO
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
    # enforce increasing r
    idx = np.argsort(r)
    return r[idx], c2[idx], c4[idx]


def parse_L_from_filename(fname: str) -> Optional[int]:
    m = re.search(r"_L([0-9]+)_", os.path.basename(fname))
    return int(m.group(1)) if m else None


# ----------------------------
# Physics helpers
# ----------------------------
def compute_tail_structure_factors(
    r: np.ndarray,
    c: np.ndarray,
    L: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays (r0_values, S_tail(r0)).
    Uses |C(r)| and weight (L-r).
    """
    rr = r.astype(int)
    cc = np.abs(c).astype(float)

    # r0 will be every r present in file (typically 1..L/2)
    r0_vals = rr.copy()
    S_tail = np.zeros_like(r0_vals, dtype=float)

    # precompute weights and cumulative sums from the end
    w = (L - rr).astype(float)
    term = w * cc

    # cumulative from end: cum_term[k] = sum_{j>=k} term[j]
    cum_term = np.cumsum(term[::-1])[::-1]

    # map r0 -> index
    # since r is typically contiguous starting at 1, index is direct, but keep general
    for i, r0 in enumerate(r0_vals):
        # first index where r >= r0
        k = int(np.searchsorted(rr, r0, side="left"))
        if k >= len(rr):
            S_tail[i] = np.nan
        else:
            S_tail[i] = (1.0 / float(L)) * cum_term[k]

    return r0_vals.astype(float), S_tail


def local_eta_from_sliding_fit(
    r: np.ndarray,
    c: np.ndarray,
    fit_window: int,
    eta_floor: float,
    r_min: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding fit of ln|C| vs ln r on a window of points.
    Trusted window (A): require ALL |C| in window >= eta_floor and r>=r_min.
    Output x positions are the center r of each valid window.
    """
    if fit_window < 3:
        raise ValueError("fit_window must be >= 3")
    if fit_window % 2 == 0:
        # keep deterministic: enforce odd
        fit_window += 1

    rr = r.astype(float)
    cc = np.abs(c).astype(float)

    # only consider r >= r_min
    mask_rmin = rr >= float(r_min)
    rr = rr[mask_rmin]
    cc = cc[mask_rmin]

    if len(rr) < fit_window:
        return np.array([], dtype=float), np.array([], dtype=float)

    half = fit_window // 2
    x_out = []
    eta_out = []

    ln_r = np.log(rr)

    for center in range(half, len(rr) - half):
        sl = slice(center - half, center + half + 1)
        cwin = cc[sl]
        if np.any(~np.isfinite(cwin)):
            continue

        # Trusted window (A)
        if np.any(cwin < float(eta_floor)):
            continue

        y = np.log(cwin)
        x = ln_r[sl]

        # least squares fit y = a + b x
        # b = cov(x,y)/var(x)
        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = float(np.dot(x0, x0))
        if denom <= 0.0:
            continue
        b = float(np.dot(x0, y0) / denom)

        eta = -b
        x_out.append(rr[center])
        eta_out.append(eta)

    return np.array(x_out, dtype=float), np.array(eta_out, dtype=float)


# ----------------------------
# Plot
# ----------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--corr_csv", type=str, required=True, help="correlators_*.csv file for a chosen (U,L,chi)")
    ap.add_argument("--L", type=int, default=None, help="System size L. If omitted, try parse from filename.")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--out_prefix", type=str, default="dmrg_quartet")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--dpi", type=int, default=800)

    # A + B2 controls
    ap.add_argument("--fit_window", type=int, default=7, help="Odd window size for sliding log-log fit")
    ap.add_argument("--eta_floor", type=float, default=1e-14, help="Trusted-window floor for eta fit: require |C|>=eta_floor in window")
    ap.add_argument("--r_min", type=int, default=2, help="Minimum r used for eta fits (avoid r=1 artifacts)")
    ap.add_argument("--eta_ylim", type=str, default=None, help="Optional y-lims for eta panel: ymin,ymax")

    # Tail ratio controls
    ap.add_argument("--tail_floor", type=float, default=1e-18, help="Mask R_tail where S2_tail < tail_floor (avoid blow-ups)")
    ap.add_argument("--r0_min", type=int, default=1, help="Minimum r0 shown in tail panel")
    ap.add_argument("--r0_max", type=int, default=None, help="Maximum r0 shown in tail panel (default: max available)")

    # Style
    ap.add_argument("--labelsize", type=float, default=18)
    ap.add_argument("--ticksize", type=float, default=14)
    ap.add_argument("--legendsize", type=float, default=12)
    ap.add_argument("--markersize", type=float, default=4.5)
    ap.add_argument("--linewidth", type=float, default=1.6)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    r, c2, c4 = read_correlators_csv(args.corr_csv)

    L = args.L
    if L is None:
        Lp = parse_L_from_filename(args.corr_csv)
        if Lp is None:
            raise RuntimeError("Could not infer L from filename; pass --L explicitly.")
        L = Lp

    # --- eta (A+B2) ---
    r_eta2, eta2 = local_eta_from_sliding_fit(r, c2, args.fit_window, args.eta_floor, args.r_min)
    r_eta4, eta4 = local_eta_from_sliding_fit(r, c4, args.fit_window, args.eta_floor, args.r_min)

    # --- R_tail ---
    r0_2, S2_tail = compute_tail_structure_factors(r, c2, L=L)
    r0_4, S4_tail = compute_tail_structure_factors(r, c4, L=L)
    assert np.allclose(r0_2, r0_4)
    r0 = r0_2

    R_tail = np.full_like(r0, np.nan, dtype=float)
    good = np.isfinite(S2_tail) & np.isfinite(S4_tail) & (S2_tail >= float(args.tail_floor))
    R_tail[good] = S4_tail[good] / S2_tail[good]

    # r0 range
    m_r0 = (r0 >= float(args.r0_min))
    if args.r0_max is not None:
        m_r0 &= (r0 <= float(args.r0_max))

    # --- plot ---
    fig = plt.figure(figsize=(7.6, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    # top: eta(r)
    if len(r_eta2) > 0:
        ax_top.plot(r_eta2, eta2, marker="o", markersize=args.markersize,
                    linewidth=args.linewidth, label=r"$\eta_{2e}(r)$")
    if len(r_eta4) > 0:
        ax_top.plot(r_eta4, eta4, marker="s", markersize=args.markersize,
                    linewidth=args.linewidth, linestyle="--", label=r"$\eta_{4e}(r)$")

    ax_top.set_xlabel(r"$r$ (rungs, $a=1$)", fontsize=args.labelsize, labelpad=2)
    ax_top.set_ylabel(r"$\eta(r)$", fontsize=args.labelsize)
    ax_top.tick_params(axis="both", labelsize=args.ticksize)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(fontsize=args.legendsize, frameon=True, loc="best")
    ax_top.margins(x=0.04, y=0.08)

    if args.eta_ylim is not None:
        y0, y1 = [float(x.strip()) for x in args.eta_ylim.split(",")]
        ax_top.set_ylim(y0, y1)

    # bottom: R_tail(r0)
    ax_bot.plot(r0[m_r0], R_tail[m_r0], marker="o", markersize=args.markersize,
                linewidth=args.linewidth, label=rf"$R_{{\mathrm{{tail}}}}(L; r_0)$")
    ax_bot.axhline(1.0, linewidth=args.linewidth, linestyle=":", label=r"$R=1$")
    ax_bot.set_xlabel(r"$r_0$ (rungs, $a=1$)", fontsize=args.labelsize)
    ax_bot.set_ylabel(r"$R_{\mathrm{tail}}$", fontsize=args.labelsize)
    ax_bot.tick_params(axis="both", labelsize=args.ticksize)
    ax_bot.grid(True, alpha=0.25)
    ax_bot.legend(fontsize=args.legendsize, frameon=True, loc="best")
    ax_bot.margins(x=0.04, y=0.10)

    # Title (small, optional)
    #fig.suptitle(rf"$L={L}$, window={args.fit_window}, $\eta$-floor={args.eta_floor:g}", fontsize=args.ticksize, y=0.98)

    out_pdf = os.path.join(args.out_dir, f"fig_eta_Rtail_2panel_L{L}_{args.out_prefix}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    if args.png:
        out_png = os.path.splitext(out_pdf)[0] + ".png"
        fig.savefig(out_png, bbox_inches="tight", dpi=int(args.dpi))
    plt.close(fig)

    print(f"[OK] wrote {out_pdf}")
    print("[INFO] eta ranges:",
          f"2e: {len(r_eta2)} pts, r in [{r_eta2.min():.0f},{r_eta2.max():.0f}]" if len(r_eta2) else "2e: none",
          f" | 4e: {len(r_eta4)} pts, r in [{r_eta4.min():.0f},{r_eta4.max():.0f}]" if len(r_eta4) else "4e: none")
    if len(r_eta2) == 0 or len(r_eta4) == 0:
        print("[HINT] If eta stops too early, try smaller --eta_floor (e.g. 1e-16) and/or smaller --fit_window (e.g. 5).")


if __name__ == "__main__":
    main()
