# Reproducibility guide

## 1. Lightweight validation

```bash
conda env create -f environment-pyscf.yml
conda activate pc60-pyscf
make check
python 02_pyscf_static/00_frontier_analysis.py --basis sto-3g --threads 4
```

## 2. Production environments

Three environments are recommended because Libra and TENSO have specialized compiled dependencies:

- `PYSCF_PYTHON`: Python executable containing PySCF, PySCF-dispersion, NumPy, SciPy, and geomeTRIC.
- `LIBRA_PYTHON`: Python executable containing `liblibra_core`, NumPy, and SciPy.
- `TENSO_PYTHON`: Python executable containing TENSO and its tensor-network dependencies.

Export those variables before `sbatch`, or replace them with site-specific module/conda activation in the Slurm files. No repository file contains a username or absolute project directory.

## 3. Full run order

From `03_aimd/`:

```bash
sbatch slurm/01_aimd_array.slurm
sbatch slurm/02_electronic_array.slurm
sbatch slurm/03_tracking_array.slurm
sbatch slurm/04_libra_array.slurm

python scripts/05_analyze_fssh.py --root output_libra/fssh --ntraj 10
python scripts/06_build_reduced_model.py --root output_tracked --ntraj 10
python scripts/07_prepare_pca_bath.py --root output_reduced_4d3a --ntraj 10 --nmodes 18
python scripts/08_fit_bath_correlations.py --nmodes 12
sbatch slurm/05_tenso.slurm
```

For the Boltzmann sensitivity calculation:

```bash
BOLTZMANN=1 OUTROOT=output_libra/fssh_boltzmann \
  sbatch slurm/04_libra_array.slurm
```

Final comparison:

```bash
python scripts/10_plot_libra_experiment.py \
  --libra-root output_libra/fssh \
  --tracked-root output_tracked \
  --experiment experiment_shg_digitized.csv \
  --outdir output_libra/final_figures --ntraj 10

python scripts/11_compare_fssh_variants.py \
  --plain-root output_libra/fssh \
  --boltzmann-root output_libra/fssh_boltzmann \
  --tracked-root output_tracked \
  --experiment experiment_shg_digitized.csv \
  --outdir output_libra/final_figures_boltzmann --ntraj 10
```

## 4. Randomness and metadata

Replica seeds and starting structures are fixed in the Slurm files and `production_starts/manifest.csv`. FSSH seeds are deterministic functions of the replica index. Each stage writes its command-line settings and diagnostics to JSON/CSV alongside numerical arrays.

## 5. Data not stored in Git

Raw AIMD trajectories, 2,000 electronic frame archives, checkpoints, full NPZ arrays, and TENSO working tensors are excluded because of size. Regeneration is computationally expensive but deterministic from the stored starting structures, seeds, scripts, and parameters. Compact headline values are under `results/`.
