# Quaternionic Superconductivity

Code, data, and figure scripts accompanying the manuscript **Quaternionic superconductivity links spinful pairing, topology, and charge-4e order**. The repository contains a spinful Bogoliubov–de Gennes (BdG) example, prescribed vortex and Josephson illustrations, and finite-chain DMRG calculations with TeNPy.

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate tenpy
```

The figure scripts use NumPy, pandas, and matplotlib. DMRG requires TeNPy (`physics-tenpy`); MPI scans also require `mpi4py`. See `environment.yml` for the environment used by the project.

## Repository layout

- `Figures/`: exported manuscript figures and optional diagnostics.
- `DMRG/quartet_model_tenpy.py`: spinful two-orbital chain model.
- `DMRG/run_dmrg_quartet_mpi.py`: DMRG driver, parallelized across scan points.
- `DMRG/results/`: stored correlators, structure factors, and run logs.
- `make_fig*.py`: figure-generation scripts described below.
- `plot_eta_Rtail_2panel.py`: local exponents and weighted tail ratios.
- `plot_chi_convergence_correlators_SM.py`: bond-dimension comparison used in Appendix C.
- `tests/test_bdg.py`: BdG symmetry and analytic-limit checks.

Plots use STIX fonts, high-resolution PNG, and PDF output. The BdG plots show energies divided by the hopping $t$; their CSV files store energies $E$. Momenta are in lattice units and the default hopping is $t=1$.

## Figure 1: full BdG excitation gap

```bash
python make_fig1_fig2.py --figure 1 --out-dir Figures
```

The script diagonalizes the full spinful Hamiltonian in the Kramers-hole basis:

$$H'(\mathbf{k})=\begin{pmatrix}h(\mathbf{k})&M(\mathbf{k})\\M(\mathbf{k})&-h(\mathbf{k})\end{pmatrix},$$

where $h=\xi I+\alpha(\sin k_y\,\sigma_x-\sin k_x\,\sigma_y)$,
$M=\Delta_s I+\Delta_t(\sin k_x\,\sigma_x+\sin k_y\,\sigma_y)$, and
$\xi=-2t(\cos k_x+\cos k_y)-\mu$.

The plotted quantity is $E_{\rm gap}(\mathbf{k})/t=\min_n|E_n(\mathbf{k})|/t$, including spin–orbit coupling and mixed-pairing branch splitting. The default $501\times501$ mesh uses $(t,\mu,\alpha,\Delta_s,\Delta_t)=(1,-2,0.6,0.25,0.35)$ and gives a minimum gap of approximately $0.20305822t$.

These parameters describe a gapped, topologically trivial class-DIII state ($\nu=0$), as established by the gap-preserving deformation in Appendix B. A gap map alone does not determine a topological invariant.

Outputs: `Figures/fig1_bdg_gapmap_nt_hd.png`, `.pdf`, and `.csv` (columns `kx, ky, Egap`).

## Figure 2: prescribed quartet vortex

```bash
python make_fig1_fig2.py --figure 2 --out-dir Figures
```

The profile is $|Q|=0.42\tanh[r/(\sqrt{2}\,8)]$ with phase $\mathrm{atan2}(y-y_0,x-x_0)$ on a $256\times256$ grid. Coordinates are in lattice units, with the vortex centered at fractions $(0.50,0.56)$ of the grid extent. This illustrates quartet winding; it is not a numerical GL minimization or a calculation of magnetic flux.

Outputs: `fig2_Q_mag_nt_cb_hd` and `fig2_Q_phase_nt_cb_hd`, each as PNG, PDF, and CSV in `Figures/`. Omit `--figure` to generate both Figures 1 and 2. Omitting `--out-dir` writes to the current working directory.

## Figures 3–5: phenomenological Josephson illustrations

These scripts write to the current working directory. To place their outputs in `Figures/`, run:

```bash
cd Figures
python ../make_fig3_ac_and_cpr.py
python ../make_fig4_squid_ic_flux.py
python ../make_fig5_Shapiro.py
cd ..
```

- **Figure 3**, `fig3_CPR_nt2_hd` and `fig3_AC_emission_nt2_hd`: the prescribed current–phase relation uses $I_1=0.12$, $I_2=1$; the AC panel compares normalized first and second harmonics. These amplitudes are not extracted from BdG or DMRG. CSV files are written under `csv/` in the working directory.
- **Figure 4**, `fig4_SQUID_Ic_flux_nt_hd`: the critical current is maximized over phase for imposed CPR amplitudes $(I_1,I_2)=(1,0),(0,1),(1,0.5)$. Flux is expressed in units of $\Phi_0=h/(2e)$.
- **Figure 5**, `fig5_Shapiro_steps_nt`: a schematic of integer and half-integer voltage positions in units of $hf/(2e)$. It does not solve driven junction dynamics or predict step widths.

All three scripts export PNG, PDF, and CSV files. Their signatures illustrate coherent two-pair transport; they are not independent evidence for a bulk charge-$4e$ condensate.

## Optional strip diagnostic (legacy Figure S1)

```bash
python make_figS1_edge_spectrum_nt.py --out-dir Figures
```

The strip uses 60 open y sites and 101 x momenta, with the same parameters as Figure 1. In the conventional Nambu basis its hole block is $-h^{\mathsf T}(-k_x)$, including momentum reversal. The CSV retains all eigenvalues; only the plot is restricted to $-2\le E/t\le2$.

Outputs: `figS1_edge_spectrum_nt_exact_hd.png`, `.pdf`, and `.csv`. There is no protected helical crossing for these parameters. The revised manuscript uses the analytic argument in Appendix B instead of this plot; the legacy filename is retained for reproducibility.

## DMRG and Appendix C

The stored comparison uses $L=80$, $U=4$, $t=1$, $t_\perp=0.6$, $\mu=-1.2$, $J_p=0$, fixed particle number $N=160$, and bond dimensions $\chi=800,1000,1200$. Its finite-distance results retain bond-dimension dependence and do not establish a thermodynamic charge-$4e$ phase.

To reproduce the bond-dimension comparison from the repository root:

```bash
python plot_chi_convergence_correlators_SM.py \
  --data_dir DMRG/results --out_dir Figures \
  --U 4.0 --L 80 --chis 800,1000,1200 \
  --rmax 40 --floor 1e-16 --png --dpi 800
```

Output: `fig_chi_convergence_corr_U4.0_L80_dmrg_quartet.(png|pdf)`.

To reproduce the exponent and full-tail diagnostic:

```bash
python plot_eta_Rtail_2panel.py \
  --corr_csv DMRG/results/correlators_dmrg_quartet_L80_t1.0_tp0.6_U4.0_mu-1.2_Jp0.0_N160_chi800.csv \
  --out_dir Figures --out_prefix U4.0_chi800 \
  --fit_window 7 --eta_floor 1e-14 --r_min 2 \
  --tail_floor 1e-18 --r0_min 1 --png --dpi 800
```

Output: `fig_eta_Rtail_2panel_L80_U4.0_chi800.(png|pdf)`.
The exponent fit requires every point in each seven-point window to exceed the floor. The tail diagnostic uses the available tail with weights $L-r$ and a $10^{-18}$ denominator floor. The common-window $r=10\ldots15$ comparison discussed in Appendix C is distinct from that full-tail plot.

To rerun the DMRG calculations, limit BLAS threading when using MPI:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
mpirun -np 3 python DMRG/run_dmrg_quartet_mpi.py \
  --Ls 80 --Us 4.0 --chis 800,1000,1200 \
  --t 1.0 --t_perp 0.6 --mu -1.2 \
  --conserve N --N_target 160 \
  --out_dir DMRG/results --out_prefix dmrg_quartet
```

Rerunning the driver replaces outputs with matching names; use another output directory to retain the stored runs.

## Numerical checks

```bash
python -m unittest discover -s tests -v
```

The seven BdG tests check Hermiticity, particle-hole and time-reversal symmetry, pairing antisymmetry, the conventional/Kramers basis transformation, analytic singlet and zero-SOC limits, and retention of energies outside the strip plot window. They do not rerun DMRG.

## Citation

If you use or adapt this repository for publication, cite the associated manuscript and acknowledge TeNPy for DMRG calculations where applicable.
