"""Basic sanity tests for the proDA pipeline — no R needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyproda import (
    fit_dropout_curves,
    fit_one_protein,
    invprobit,
    proda,
)
from pyproda import test_diff as run_test_diff


def _make_dataset(seed: int = 42, n_proteins: int = 300, n_per_group: int = 6,
                  rho_true: float = 18.0, zeta_true: float = 0.5,
                  prop_de: float = 0.15, effect_size: float = 1.0):
    """Synthetic LFQ-style log2 matrix with MNAR-censored low values."""
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_per_group
    base = rng.normal(20.0, 1.5, n_proteins)
    sigma = 0.5 + 0.5 * rng.uniform(size=n_proteins)
    delta = np.zeros(n_proteins)
    is_de = rng.uniform(size=n_proteins) < prop_de
    delta[is_de] = rng.choice([-1, 1], is_de.sum()) * effect_size
    M = np.zeros((n_proteins, n_samples))
    for g in [0, 1]:
        mean_g = base + (delta if g else 0.0)
        M[:, g * n_per_group:(g + 1) * n_per_group] = rng.normal(
            mean_g[:, None], sigma[:, None], (n_proteins, n_per_group),
        )
    # Probit-style MNAR censoring: P(observed | y) = Phi((y - rho)/zeta).
    p_obs = invprobit(M, rho_true, zeta_true)
    M_mis = rng.uniform(size=M.shape) > p_obs
    M[M_mis] = np.nan
    M_df = pd.DataFrame(
        M,
        index=[f"prot_{i:04d}" for i in range(n_proteins)],
        columns=[f"S{j:02d}" for j in range(n_samples)],
    )
    design = ["A"] * n_per_group + ["B"] * n_per_group
    return M_df, design, is_de, M_mis


def test_invprobit_matches_normal_cdf():
    x = np.linspace(15, 22, 20)
    out = invprobit(x, 18.0, 0.5)
    from scipy.stats import norm
    expected = norm.cdf((x - 18.0) / 0.5)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_invprobit_oneminus_and_log():
    x = np.linspace(15, 22, 20)
    p = invprobit(x, 18.0, 0.5)
    p_log = invprobit(x, 18.0, 0.5, log=True)
    p_one = invprobit(x, 18.0, 0.5, oneminus=True)
    np.testing.assert_allclose(np.exp(p_log), p, atol=1e-12)
    np.testing.assert_allclose(p_one, 1.0 - p, atol=1e-12)


def test_fit_one_protein_no_missing_recovers_OLS():
    rng = np.random.default_rng(0)
    n, p = 12, 2
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta_true = np.array([5.0, 2.5])
    y = X @ beta_true + rng.normal(0, 0.3, size=n)
    fit = fit_one_protein(y, X)
    np.testing.assert_allclose(fit.coefficients, beta_true, atol=0.2)
    assert fit.n_obs == n
    assert fit.df == n - p
    assert fit.s2 > 0


def test_fit_one_protein_with_missing_uses_censored_mle():
    rng = np.random.default_rng(1)
    n = 12
    X = np.column_stack([np.ones(n), np.array([0]*6 + [1]*6)])
    beta_true = np.array([18.5, 1.0])
    y = X @ beta_true + rng.normal(0, 0.4, size=n)
    y[:3] = np.nan          # drop some low-intensity samples
    rho = np.full(n, 18.0)
    zeta = np.full(n, 0.5)
    fit = fit_one_protein(y, X, rho=rho, zeta=zeta)
    assert fit.n_obs == 9
    assert np.all(np.isfinite(fit.coefficients))
    assert np.all(np.isfinite(np.diag(fit.coef_variance_matrix)))


def test_dropout_curves_recover_truth_roughly():
    rng = np.random.default_rng(7)
    n_proteins, n_samples = 200, 8
    Pred = rng.normal(20.0, 1.0, (n_proteins, n_samples))
    Y = Pred + rng.normal(0, 0.5, (n_proteins, n_samples))
    rho_true, zeta_true = 19.0, 0.4
    p_obs = invprobit(Y, rho_true, zeta_true)
    mask = rng.uniform(size=Y.shape) > p_obs
    Y[mask] = np.nan
    hp = fit_dropout_curves(Y, Pred)
    # Median across samples should be within 0.5 of truth.
    assert abs(float(np.nanmedian(hp.rho)) - rho_true) < 0.5
    assert 0.1 < float(np.nanmedian(hp.zeta)) < 1.5


def test_proda_pipeline_runs_end_to_end():
    M, design, is_de, _ = _make_dataset()
    fit = proda(M, design=design, max_iter=3)
    assert fit.coefficients.shape == (M.shape[0], 2)
    assert len(fit.coef_variance_matrices) == M.shape[0]
    assert fit.hyper_parameters.rho.size == M.shape[1]


def test_test_diff_returns_table_with_correct_columns():
    M, design, is_de, _ = _make_dataset()
    fit = proda(M, design=design, max_iter=2)
    res = run_test_diff(fit, np.array([0.0, 1.0]))
    for col in ("name", "pval", "adj_pval", "diff", "t_statistic",
                "se", "df", "avg_abundance", "n_obs"):
        assert col in res.columns
    assert len(res) == M.shape[0]
    # True DE proteins should be enriched among low-pval ones.
    pval_de = res.set_index("name").loc[
        M.index[is_de].astype(str), "pval"
    ]
    pval_not = res.set_index("name").loc[
        M.index[~is_de].astype(str), "pval"
    ]
    assert np.nanmean(pval_de) < np.nanmean(pval_not)
