"""Head-to-head speed+accuracy benchmark: R proDA vs pyproda.

Runs ``proDA(M, design) -> test_diff(fit, contrast)`` on a synthetic
300-protein x 12-sample MNAR-censored dataset and reports:

* wall-clock time for the full Python pipeline (mean of ``--runs``).
* wall-clock time for the R pipeline (single ``Rscript`` run; the
  ``timing.tsv`` file also gives the in-R per-step breakdown).
* MAE + Pearson r between Python and R for: per-protein coefficients,
  Wald ``t_statistic``, Wald ``pval``.

The synthetic matrix is generated deterministically in NumPy and dumped
to TSV so both languages see identical input.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pyproda import invprobit, proda
from pyproda import test_diff as run_test_diff


HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
R_DRIVER = HERE.parent / "tests" / "r_reference_driver.R"


def _make_dataset(
    n_proteins: int, n_per_group: int, n_groups: int = 2, seed: int = 42,
    rho_true: float = 18.0, zeta_true: float = 0.5,
):
    rng = np.random.default_rng(seed)
    n_samples = n_per_group * n_groups
    base = rng.normal(20.0, 1.5, n_proteins)
    sigma = 0.5 + 0.5 * rng.uniform(size=n_proteins)
    delta = np.zeros(n_proteins)
    is_de = rng.uniform(size=n_proteins) < 0.15
    delta[is_de] = rng.choice([-1, 1], is_de.sum()) * 1.0
    M = np.zeros((n_proteins, n_samples))
    for g in range(n_groups):
        mean_g = base + (delta if g == 1 else 0.0)
        M[:, g * n_per_group:(g + 1) * n_per_group] = rng.normal(
            mean_g[:, None], sigma[:, None], (n_proteins, n_per_group),
        )
    p_obs = invprobit(M, rho_true, zeta_true)
    M_mis = rng.uniform(size=M.shape) > p_obs
    M[M_mis] = np.nan
    M_df = pd.DataFrame(
        M,
        index=[f"prot_{i:05d}" for i in range(n_proteins)],
        columns=[f"S{j:02d}" for j in range(n_samples)],
    )
    design_df = pd.DataFrame(
        {"group": ["A"] * n_per_group + ["B"] * n_per_group},
        index=M_df.columns,
    )
    return M_df, design_df, is_de


def run_R(work):
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} && Rscript "
        f"{R_DRIVER} {work / 'M.tsv'} {work / 'design.tsv'} {work / 'R_out'}"
    )
    t0 = time.perf_counter()
    subprocess.run(["bash", "-lc", cmd], check=True,
                   capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    timing = pd.read_csv(work / "R_out" / "timing.tsv", sep="\t")
    return elapsed, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--n-proteins", type=int, default=300)
    ap.add_argument("--n-per-group", type=int, default=6)
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    M_df, design_df, is_de = _make_dataset(
        args.n_proteins, args.n_per_group,
    )
    M_df.to_csv(WORK / "M.tsv", sep="\t", na_rep="NA",
                index_label="protein")
    design_df.to_csv(WORK / "design.tsv", sep="\t", index_label="sample")

    n_miss = int(np.isnan(M_df.to_numpy()).sum())
    print(
        f"Matrix: {M_df.shape[0]} proteins × {M_df.shape[1]} samples "
        f"({n_miss / M_df.size:.1%} missing)"
    )

    print(f"\n--- Python timing (mean of {args.runs} runs) ---")
    py_total = []
    res_py = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        fit = proda(M_df, design=design_df["group"].tolist(), max_iter=5)
        res_py = run_test_diff(fit, np.array([0.0, 1.0]))
        py_total.append(time.perf_counter() - t0)
    py_total_mean = float(np.mean(py_total))
    print(f"  py total           {py_total_mean*1000:9.2f} ms")

    print("\n--- R timing (single run; per-step breakdown) ---")
    r_total, r_timing = run_R(WORK)
    print(r_timing.to_string(index=False))
    r_pipeline = float(r_timing.sum(axis=1).iloc[0])
    print(f"  R proda+test_diff  {r_pipeline*1000:9.2f} ms")
    print(f"  R total wall       {r_total*1000:9.2f} ms (incl. Rscript startup)")

    print(f"\nSpeedup (R pipeline / Python pipeline): {r_pipeline / py_total_mean:.2f}x")

    # Accuracy comparison.
    r_coefs = pd.read_csv(WORK / "R_out" / "coefs.tsv",
                          sep="\t", index_col=0)
    r_td = pd.read_csv(WORK / "R_out" / "test_diff.tsv", sep="\t")
    py_coefs = fit.coefficients  # uses last python fit
    py_idx = res_py.set_index("name")
    py_aligned = py_idx.reindex(r_td["name"])

    # Coef comparisons (per coefficient column).
    coef_metrics = {}
    for j, name in enumerate(fit.coef_names):
        rc = r_coefs.iloc[:, j].to_numpy()
        pc = py_coefs[:, j]
        mask = np.isfinite(rc) & np.isfinite(pc)
        mae = float(np.mean(np.abs(rc[mask] - pc[mask])))
        r = float(np.corrcoef(rc[mask], pc[mask])[0, 1])
        coef_metrics[name] = {"mae": mae, "pearson_r": r}

    py_t = py_aligned["t_statistic"].to_numpy()
    r_t = r_td["t_statistic"].to_numpy()
    mask_t = np.isfinite(py_t) & np.isfinite(r_t)
    t_mae = float(np.mean(np.abs(py_t[mask_t] - r_t[mask_t])))
    t_r = float(np.corrcoef(py_t[mask_t], r_t[mask_t])[0, 1])

    py_p = py_aligned["pval"].to_numpy()
    r_p = r_td["pval"].to_numpy()
    mask_p = np.isfinite(py_p) & np.isfinite(r_p)
    p_mae = float(np.mean(np.abs(py_p[mask_p] - r_p[mask_p])))
    p_r = float(np.corrcoef(py_p[mask_p], r_p[mask_p])[0, 1])

    # Top-N DE agreement.
    top_n = max(int(args.n_proteins * 0.15), 30)
    r_top = set(r_td.nsmallest(top_n, "pval")["name"])
    py_top = set(res_py.nsmallest(top_n, "pval")["name"])
    overlap = len(r_top & py_top) / top_n

    print("\n--- Accuracy (Python vs R) ---")
    for name, m in coef_metrics.items():
        print(f"  coef[{name}]  MAE={m['mae']:.4g}  Pearson r={m['pearson_r']:.4f}")
    print(f"  t-stat    MAE={t_mae:.4g}  Pearson r={t_r:.4f}")
    print(f"  pval      MAE={p_mae:.4g}  Pearson r={p_r:.4f}")
    print(f"  top-{top_n} agreement: {overlap:.1%}")

    summary = {
        "shape": [M_df.shape[0], M_df.shape[1]],
        "py_time_ms": py_total_mean * 1000,
        "r_time_ms_pipeline": r_pipeline * 1000,
        "r_time_ms_total": r_total * 1000,
        "speedup": r_pipeline / py_total_mean,
        "coef_metrics": coef_metrics,
        "t_stat_mae": t_mae,
        "t_stat_pearson_r": t_r,
        "pval_mae": p_mae,
        "pval_pearson_r": p_r,
        "top_overlap": overlap,
        "top_n": top_n,
    }
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report -> {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
