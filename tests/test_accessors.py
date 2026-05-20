"""Smoke tests for the R-style functional accessors on ``ProDAFit``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyproda import (
    abundances,
    coefficient_variance_matrices,
    coefficients,
    convergence,
    design,
    feature_parameters,
    hyper_parameters,
    invprobit,
    predict,
    proda,
    reference_level,
    result_names,
)


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(0)
    n_p, n_s = 80, 10
    M = rng.normal(20.0, 1.5, (n_p, n_s))
    p_obs = invprobit(M, 18.0, 0.5)
    M[rng.uniform(size=M.shape) > p_obs] = np.nan
    M_df = pd.DataFrame(
        M,
        index=[f"prot_{i:03d}" for i in range(n_p)],
        columns=[f"S{j:02d}" for j in range(n_s)],
    )
    design_v = ["A"] * 5 + ["B"] * 5
    return proda(M_df, design=design_v, max_iter=3)


def test_abundances_shape_and_type(fit):
    ab = abundances(fit)
    assert isinstance(ab, pd.DataFrame)
    assert ab.shape == fit.data.shape
    # NaN positions are preserved.
    assert np.array_equal(np.isnan(ab.to_numpy()), np.isnan(fit.data))


def test_coefficients_is_dataframe(fit):
    cf = coefficients(fit)
    assert isinstance(cf, pd.DataFrame)
    assert cf.shape == (len(fit.protein_names), len(fit.coef_names))
    assert list(cf.columns) == fit.coef_names


def test_coefficient_variance_matrices(fit):
    cvm = coefficient_variance_matrices(fit)
    assert isinstance(cvm, list)
    assert len(cvm) == len(fit.protein_names)
    p = len(fit.coef_names)
    for m in cvm:
        assert m is None or m.shape == (p, p)


def test_convergence_dict(fit):
    conv = convergence(fit)
    assert isinstance(conv, dict)
    assert "iterations" in conv
    assert "successful" in conv


def test_design_dataframe(fit):
    d = design(fit)
    assert isinstance(d, pd.DataFrame)
    assert d.shape == fit.design.shape
    assert list(d.columns) == fit.coef_names


def test_feature_parameters_columns(fit):
    fp = feature_parameters(fit)
    assert isinstance(fp, pd.DataFrame)
    for col in ("n_approx", "df", "s2", "n_obs"):
        assert col in fp.columns
    assert len(fp) == len(fit.protein_names)


def test_hyper_parameters_dict(fit):
    hp = hyper_parameters(fit)
    assert isinstance(hp, dict)
    for key in (
        "location_prior_mean", "location_prior_scale", "location_prior_df",
        "variance_prior_scale", "variance_prior_df",
        "dropout_curve_position", "dropout_curve_scale",
    ):
        assert key in hp
    assert hp["dropout_curve_position"].size == len(fit.sample_names)
    assert hp["dropout_curve_scale"].size == len(fit.sample_names)


def test_reference_level(fit):
    # Default: no explicit reference => None.
    assert reference_level(fit) is None


def test_result_names(fit):
    rn = result_names(fit)
    assert rn == fit.coef_names


def test_predict_response(fit):
    pr = predict(fit, type="response")
    assert isinstance(pr, pd.DataFrame)
    assert pr.shape == fit.data.shape
    # Predicted matrix == coefs @ X.T.
    expected = fit.coefficients @ fit.design.T
    np.testing.assert_allclose(pr.to_numpy(), expected, atol=1e-10)


def test_predict_link_equals_response(fit):
    np.testing.assert_allclose(
        predict(fit, type="link").to_numpy(),
        predict(fit, type="response").to_numpy(),
        atol=1e-12,
    )


def test_predict_bad_type_raises(fit):
    with pytest.raises(ValueError):
        predict(fit, type="garbage")
