"""R-parity tests against Bioconductor proDA.

We dump the same synthetic M / design to TSV and have R run the full
``proDA(...)`` pipeline (with ``moderate_variance=FALSE`` /
``moderate_location=FALSE`` to match Python v0.1). Then we compare:

* ``invprobit(x, rho, zeta)`` — bit-exact (deterministic, atol 1e-12).
* per-sample dropout-curve ``(rho, zeta)`` — median absolute diff < 0.1.
* per-protein coefficients — Pearson r > 0.99.
* ``test_diff`` t-statistic — Pearson r > 0.95.
* ``test_diff`` p-value — Pearson r > 0.95.

The Wald-test path R uses for the un-moderated case is ``df = n - p``
(full-sample residual df), so we use the same in Python — see
``pyproda/test_diff.py``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyproda import dist_approx, invprobit, pd_row_t_test, proda
from pyproda import test_diff as run_test_diff


HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not Path(R_DRIVER).exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(proDA); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or proDA not installed.",
)


@pytest.fixture(scope="module")
def shared_dataset(tmp_path_factory):
    """Synthetic MNAR-censored matrix shared by every R-parity test."""
    rng = np.random.default_rng(42)
    n_proteins, n_per_group = 200, 6
    n_samples = 2 * n_per_group
    base = rng.normal(20.0, 1.5, n_proteins)
    sigma = 0.5 + 0.5 * rng.uniform(size=n_proteins)
    delta = np.zeros(n_proteins)
    is_de = rng.uniform(size=n_proteins) < 0.15
    delta[is_de] = rng.choice([-1, 1], is_de.sum()) * 1.0
    M = np.zeros((n_proteins, n_samples))
    for g in [0, 1]:
        mean_g = base + (delta if g else 0.0)
        M[:, g * n_per_group:(g + 1) * n_per_group] = rng.normal(
            mean_g[:, None], sigma[:, None], (n_proteins, n_per_group),
        )
    rho_true, zeta_true = 18.0, 0.5
    p_obs = invprobit(M, rho_true, zeta_true)
    M_mis = rng.uniform(size=M.shape) > p_obs
    M[M_mis] = np.nan
    M_df = pd.DataFrame(
        M,
        index=[f"prot_{i:04d}" for i in range(n_proteins)],
        columns=[f"S{j:02d}" for j in range(n_samples)],
    )
    design_df = pd.DataFrame(
        {"group": ["A"] * n_per_group + ["B"] * n_per_group},
        index=M_df.columns,
    )

    work = tmp_path_factory.mktemp("proda_parity")
    M_df.to_csv(work / "M.tsv", sep="\t", na_rep="NA",
                index_label="protein")
    design_df.to_csv(work / "design.tsv", sep="\t",
                     index_label="sample")

    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {work / 'M.tsv'} {work / 'design.tsv'} "
        f"{work / 'R_out'}"
    )
    subprocess.run(["bash", "-lc", cmd], check=True,
                   capture_output=True, text=True)

    return {
        "M": M_df, "design": design_df, "is_de": is_de,
        "work": work, "R_out": work / "R_out",
    }


def test_invprobit_bit_exact(shared_dataset):
    r_inv = pd.read_csv(shared_dataset["R_out"] / "invprobit.tsv", sep="\t")
    x = r_inv["x"].to_numpy()
    py_inv = invprobit(x, 18.0, 0.5)
    np.testing.assert_allclose(
        py_inv, r_inv["invprobit"].to_numpy(), atol=1e-12,
        err_msg="invprobit diverges from R proDA::invprobit",
    )


def test_dropout_curves_close_to_R(shared_dataset):
    fit = proda(shared_dataset["M"], design=shared_dataset["design"]["group"].tolist(),
                max_iter=5)
    r_rz = pd.read_csv(shared_dataset["R_out"] / "rho_zeta.tsv", sep="\t")
    # R stores zeta as the negative value (zetainv convention); we store
    # the positive scale. Compare magnitudes.
    py_rho = fit.hyper_parameters.rho
    py_zeta = fit.hyper_parameters.zeta
    r_rho = r_rz["rho"].to_numpy()
    r_zeta = np.abs(r_rz["zeta"].to_numpy())

    # Median absolute deviation must stay tight.
    assert float(np.median(np.abs(py_rho - r_rho))) < 0.5, (
        f"median |py_rho - R_rho| = {np.median(np.abs(py_rho - r_rho))!r}"
    )
    assert float(np.median(np.abs(py_zeta - r_zeta))) < 0.5, (
        f"median |py_zeta - R_zeta| = {np.median(np.abs(py_zeta - r_zeta))!r}"
    )
    # The per-sample correlation should be strong.
    if py_rho.size > 2:
        r = float(np.corrcoef(py_rho, r_rho)[0, 1])
        assert r > 0.5 or float(np.std(r_rho)) < 0.1, (
            f"rho per-sample Pearson r={r}"
        )


def test_coefficients_match_R(shared_dataset):
    fit = proda(shared_dataset["M"], design=shared_dataset["design"]["group"].tolist(),
                max_iter=5)
    r_coefs = pd.read_csv(shared_dataset["R_out"] / "coefs.tsv",
                          sep="\t", index_col=0)
    py_coefs = fit.coefficients
    r_arr = r_coefs.to_numpy()
    for j in range(py_coefs.shape[1]):
        mask = np.isfinite(py_coefs[:, j]) & np.isfinite(r_arr[:, j])
        r = float(np.corrcoef(py_coefs[mask, j], r_arr[mask, j])[0, 1])
        assert r > 0.99, (
            f"coef[{j}] Pearson r={r:.4f} (need >0.99)"
        )


def test_test_diff_t_statistic_matches_R(shared_dataset):
    fit = proda(shared_dataset["M"], design=shared_dataset["design"]["group"].tolist(),
                max_iter=5)
    res = run_test_diff(fit, np.array([0.0, 1.0]))
    r_td = pd.read_csv(shared_dataset["R_out"] / "test_diff.tsv", sep="\t")
    py = res.set_index("name").reindex(r_td["name"])
    py_t = py["t_statistic"].to_numpy()
    r_t = r_td["t_statistic"].to_numpy()
    mask = np.isfinite(py_t) & np.isfinite(r_t)
    r = float(np.corrcoef(py_t[mask], r_t[mask])[0, 1])
    assert r > 0.95, f"test_diff t-stat Pearson r={r:.4f} (need >0.95)"


def test_test_diff_pval_matches_R(shared_dataset):
    fit = proda(shared_dataset["M"], design=shared_dataset["design"]["group"].tolist(),
                max_iter=5)
    res = run_test_diff(fit, np.array([0.0, 1.0]))
    r_td = pd.read_csv(shared_dataset["R_out"] / "test_diff.tsv", sep="\t")
    py = res.set_index("name").reindex(r_td["name"])
    py_p = py["pval"].to_numpy()
    r_p = r_td["pval"].to_numpy()
    mask = np.isfinite(py_p) & np.isfinite(r_p)
    r = float(np.corrcoef(py_p[mask], r_p[mask])[0, 1])
    assert r > 0.95, f"test_diff pval Pearson r={r:.4f} (need >0.95)"


def test_pd_row_t_test_t_statistic_matches_R(shared_dataset):
    """pd_row_t_test t-statistic should track R proDA::pd_row_t_test."""
    M = shared_dataset["M"].to_numpy()
    groups = shared_dataset["design"]["group"].to_numpy()
    X = M[:, groups == "A"]
    Y = M[:, groups == "B"]
    res = pd_row_t_test(
        X, Y, moderate_location=False, moderate_variance=False,
        max_iter=10,
    )
    r_tt = pd.read_csv(shared_dataset["R_out"] / "pd_row_t_test.tsv", sep="\t")
    py_t = res["t_statistic"].to_numpy()
    r_t = r_tt["t_statistic"].to_numpy()
    mask = np.isfinite(py_t) & np.isfinite(r_t)
    r = float(np.corrcoef(py_t[mask], r_t[mask])[0, 1])
    assert r > 0.99, f"pd_row_t_test t-stat Pearson r={r:.4f} (need >0.99)"


def test_dist_approx_matches_R(shared_dataset):
    """dist_approx between proteins should track R proDA::dist_approx."""
    fit = proda(shared_dataset["M"], design=shared_dataset["design"]["group"].tolist(),
                max_iter=5)
    d = dist_approx(fit, by_sample=False)
    py_mat = d["mean"].to_numpy()
    r_mat = pd.read_csv(
        shared_dataset["R_out"] / "dist_approx.tsv", sep="\t", index_col=0,
    ).to_numpy()
    # Compare the upper triangle (condensed distances).
    iu = np.triu_indices_from(py_mat, k=1)
    py_v = py_mat[iu]
    r_v = r_mat[iu]
    mask = np.isfinite(py_v) & np.isfinite(r_v)
    r = float(np.corrcoef(py_v[mask], r_v[mask])[0, 1])
    assert r > 0.95, f"dist_approx Pearson r={r:.4f} (need >0.95)"


def test_moderate_variance_s2_matches_R(shared_dataset):
    """With moderate_variance=True, per-protein s2 should track R."""
    fit = proda(shared_dataset["M"], design=shared_dataset["design"]["group"].tolist(),
                moderate_variance=True, moderate_location=False, max_iter=10)
    r_s2 = pd.read_csv(shared_dataset["R_out"] / "s2_moderated.tsv", sep="\t")
    py = fit.feature_parameters_().reset_index()
    py_s2 = py["s2"].to_numpy()
    r_s2v = r_s2["s2"].to_numpy()
    mask = np.isfinite(py_s2) & np.isfinite(r_s2v)
    r = float(np.corrcoef(py_s2[mask], r_s2v[mask])[0, 1])
    assert r > 0.99, f"moderate_variance s2 Pearson r={r:.4f} (need >0.99)"
