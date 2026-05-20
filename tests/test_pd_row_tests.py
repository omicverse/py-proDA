"""Tests for the convenience two-/multi-group row tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyproda import invprobit, pd_row_f_test, pd_row_t_test


def _two_group(seed=1, n_p=60, n_de=12, effect=2.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(20.0, 1.0, (n_p, 6))
    Y = rng.normal(20.0, 1.0, (n_p, 6))
    Y[:n_de] += effect
    for m in (X, Y):
        p_obs = invprobit(m, 18.0, 0.5)
        m[rng.uniform(size=m.shape) > p_obs] = np.nan
    return X, Y, n_de


def test_pd_row_t_test_columns():
    X, Y, _ = _two_group()
    res = pd_row_t_test(X, Y, max_iter=5)
    for col in ("name", "pval", "adj_pval", "diff", "t_statistic",
                "se", "df", "avg_abundance", "n_obs"):
        assert col in res.columns
    assert len(res) == X.shape[0]


def test_pd_row_t_test_detects_de():
    X, Y, n_de = _two_group()
    res = pd_row_t_test(X, Y, max_iter=5)
    pval = res["pval"].to_numpy()
    de_mean = np.nanmean(pval[:n_de])
    non_mean = np.nanmean(pval[n_de:])
    assert de_mean < non_mean
    # The planted-DE rows should be enriched among the smallest p-values.
    assert de_mean < 0.2


def test_pd_row_t_test_return_fit():
    X, Y, _ = _two_group()
    out = pd_row_t_test(X, Y, max_iter=5, return_fit=True)
    assert set(out.keys()) == {"fit", "test_results"}
    assert out["fit"].coefficients.shape[0] == X.shape[0]


def test_pd_row_t_test_row_count_mismatch_raises():
    X = np.random.default_rng(0).normal(size=(10, 4))
    Y = np.random.default_rng(1).normal(size=(8, 4))
    with pytest.raises(ValueError):
        pd_row_t_test(X, Y)


def test_pd_row_f_test_multi_matrix():
    rng = np.random.default_rng(2)
    n_p, n_de = 50, 10
    X = rng.normal(20.0, 1.0, (n_p, 5))
    Y = rng.normal(20.0, 1.0, (n_p, 5))
    Z = rng.normal(20.0, 1.0, (n_p, 5))
    Y[:n_de] += 2.0
    Z[:n_de] += 3.0
    for m in (X, Y, Z):
        p_obs = invprobit(m, 18.0, 0.5)
        m[rng.uniform(size=m.shape) > p_obs] = np.nan
    res = pd_row_f_test(X, Y, Z, max_iter=5)
    for col in ("name", "pval", "adj_pval", "f_statistic",
                "df1", "df2", "avg_abundance", "n_obs"):
        assert col in res.columns
    assert len(res) == n_p
    de_mean = np.nanmean(res["pval"].to_numpy()[:n_de])
    non_mean = np.nanmean(res["pval"].to_numpy()[n_de:])
    assert de_mean < non_mean


def test_pd_row_f_test_groups_form():
    rng = np.random.default_rng(3)
    n_p = 40
    data = rng.normal(20.0, 1.0, (n_p, 9))
    data[:8, 6:] += 3.0
    p_obs = invprobit(data, 18.0, 0.5)
    data[rng.uniform(size=data.shape) > p_obs] = np.nan
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    res = pd_row_f_test(data, groups=groups, max_iter=5)
    assert len(res) == n_p
    assert "f_statistic" in res.columns
    assert (res["df1"].dropna() == 2).all()  # 3 groups => p_full - 1 = 2


def test_pd_row_f_test_single_condition_raises():
    data = np.random.default_rng(0).normal(size=(10, 6))
    with pytest.raises(ValueError):
        pd_row_f_test(data, groups=["a"] * 6)


def test_pd_row_f_test_conflicting_args_raises():
    X = np.random.default_rng(0).normal(size=(10, 4))
    Y = np.random.default_rng(1).normal(size=(10, 4))
    with pytest.raises(ValueError):
        pd_row_f_test(X, Y, groups=["a", "b"] * 4)
