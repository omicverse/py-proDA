"""Tests for the empirical-Bayes moderation features in ``proda``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyproda import invprobit, proda


def _heteroscedastic_dataset(seed=3, n_p=200, n_s=10):
    """Synthetic matrix with a *wide spread* of per-protein variances —
    the regime where EB shrinkage has visible effect."""
    rng = np.random.default_rng(seed)
    base = rng.normal(20.0, 1.5, n_p)
    sigma = 0.3 + 1.2 * rng.uniform(size=n_p)
    M = rng.normal(base[:, None], sigma[:, None], (n_p, n_s))
    p_obs = invprobit(M, 18.0, 0.5)
    M[rng.uniform(size=M.shape) > p_obs] = np.nan
    return pd.DataFrame(M), ["A"] * (n_s // 2) + ["B"] * (n_s // 2)


def test_moderate_variance_shrinks_s2_spread():
    M, design = _heteroscedastic_dataset()
    fit_no = proda(M, design=design, moderate_variance=False, max_iter=4)
    fit_mod = proda(M, design=design, moderate_variance=True, max_iter=4)
    s2_no = fit_no.s2[np.isfinite(fit_no.s2)]
    s2_mod = fit_mod.s2[np.isfinite(fit_mod.s2)]
    # Variance of the per-protein s2 must shrink towards the prior.
    assert np.var(s2_mod) < np.var(s2_no)


def test_moderate_variance_sets_prior_fields():
    M, design = _heteroscedastic_dataset()
    fit = proda(M, design=design, moderate_variance=True, max_iter=4)
    assert np.isfinite(fit.df_prior)
    assert fit.df_prior > 0
    assert np.isfinite(fit.s2_prior)
    assert fit.s2_prior > 0
    # Hyper-parameters accessor exposes them.
    hp = fit.hyper_parameters_()
    assert np.isfinite(hp["variance_prior_scale"])
    assert np.isfinite(hp["variance_prior_df"])


def test_moderate_variance_off_has_nan_prior():
    M, design = _heteroscedastic_dataset()
    fit = proda(M, design=design, moderate_variance=False, max_iter=3)
    assert np.isnan(fit.df_prior)
    assert np.isnan(fit.s2_prior)


def test_moderate_location_sets_location_prior():
    M, design = _heteroscedastic_dataset()
    fit = proda(M, design=design, moderate_location=True, max_iter=4)
    assert np.isfinite(fit.location_prior_mean)
    assert np.isfinite(fit.location_prior_scale)
    assert fit.location_prior_df == 3.0


def test_n_subsample_chunked_fit_runs():
    M, design = _heteroscedastic_dataset(n_p=150)
    fit = proda(M, design=design, n_subsample=60, max_iter=3)
    # Every protein still gets coefficients (sub + rest stitched).
    assert fit.coefficients.shape == (150, 2)
    # Most rows fit successfully.
    finite_frac = np.isfinite(fit.coefficients).all(axis=1).mean()
    assert finite_frac > 0.8


def test_n_subsample_full_equals_no_subsample():
    M, design = _heteroscedastic_dataset(n_p=80)
    fit_full = proda(M, design=design, max_iter=3, seed=0)
    fit_sub = proda(M, design=design, n_subsample=80, max_iter=3, seed=0)
    # n_subsample == n_proteins => identical fit.
    np.testing.assert_allclose(
        np.nan_to_num(fit_full.coefficients),
        np.nan_to_num(fit_sub.coefficients),
        atol=1e-8,
    )
