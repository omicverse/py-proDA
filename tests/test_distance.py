"""Tests for ``dist_approx`` and ``median_normalization``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyproda import dist_approx, invprobit, median_normalization, proda


@pytest.fixture(scope="module")
def fit():
    rng = np.random.default_rng(4)
    n_p, n_s = 50, 8
    base = rng.normal(20.0, 1.5, n_p)
    M = rng.normal(base[:, None], 0.6, (n_p, n_s))
    p_obs = invprobit(M, 18.0, 0.5)
    M[rng.uniform(size=M.shape) > p_obs] = np.nan
    M_df = pd.DataFrame(M)
    return proda(M_df, design=["A"] * 4 + ["B"] * 4, max_iter=3)


def test_dist_approx_protein_shape(fit):
    d = dist_approx(fit)
    assert set(d.keys()) == {"mean", "sd"}
    n = len(fit.protein_names)
    assert d["mean"].shape == (n, n)
    assert d["sd"].shape == (n, n)


def test_dist_approx_is_symmetric_with_zero_diagonal(fit):
    d = dist_approx(fit)
    mat = d["mean"].to_numpy()
    np.testing.assert_allclose(mat, mat.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-10)


def test_dist_approx_values_nonnegative_and_finite_range(fit):
    d = dist_approx(fit)
    mat = d["mean"].to_numpy()
    finite = mat[np.isfinite(mat)]
    assert (finite >= 0).all()
    # Distances should be in a sensible range for log2 intensities.
    assert finite.max() < 1e3


def test_dist_approx_by_sample(fit):
    d = dist_approx(fit, by_sample=True)
    n = len(fit.sample_names)
    assert d["mean"].shape == (n, n)
    mat = d["mean"].to_numpy()
    np.testing.assert_allclose(mat, mat.T, atol=1e-10)


def test_median_normalization_centers_columns():
    rng = np.random.default_rng(5)
    M = rng.normal(20.0, 1.0, (100, 6))
    # Add per-column offsets.
    offs = np.array([0.5, -1.0, 2.0, 0.0, -0.5, 1.5])
    M = M + offs[None, :]
    Mn, applied = median_normalization(M)
    assert applied.shape == (6,)
    # The applied offset for each column equals median(col - rowMeans)
    # computed on the *original* matrix (matches R median_normalization).
    row_means = np.nanmean(M, axis=1)
    for j in range(M.shape[1]):
        expected = np.nanmedian(M[:, j] - row_means)
        assert abs(applied[j] - expected) < 1e-10
    # The recovered offsets track the planted per-column shifts (up to a
    # global constant, since row-mean centering is shift-invariant). The
    # match is approximate — the column median of a finite sample is a
    # noisy estimator of the planted offset.
    spread = applied - applied.mean()
    planted = offs - offs.mean()
    assert float(np.corrcoef(spread, planted)[0, 1]) > 0.97


def test_median_normalization_dataframe_roundtrip():
    rng = np.random.default_rng(6)
    M = pd.DataFrame(rng.normal(20.0, 1.0, (30, 4)))
    Mn, off = median_normalization(M)
    assert isinstance(Mn, pd.DataFrame)
    assert list(Mn.columns) == list(M.columns)
    assert off.shape == (4,)


def test_median_normalization_spike_in_rows():
    rng = np.random.default_rng(7)
    M = rng.normal(20.0, 1.0, (50, 5))
    spike = np.zeros(50, dtype=bool)
    spike[:10] = True
    Mn, off = median_normalization(M, spike_in_rows=spike)
    assert Mn.shape == M.shape
    assert off.shape == (5,)
