"""Probabilistic distance between proteins/samples (``dist_approx``).

Port of ``proDA::dist_approx(fit, by_sample=...)``. For each pair of
proteins (or samples) the squared Euclidean distance is approximated
under the proDA dropout model:

* observed positions contribute the raw squared difference,
* missing positions are imputed by the linear predictor ``Pred[i,j]``,
* the uncertainty of the imputation is propagated via ``Pred_var``
  (the diagonal of ``X Var(beta) X'``).

The R implementation (``proDA:::dist_approx_mean_var``) returns the
mean and variance of the squared distance; we expose the
``sqrt(mean)`` as the *expected* distance (a clean drop-in for
:func:`scipy.spatial.distance.pdist`-style outputs) and the propagated
SD as a companion array.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _distance_sq(mu1, var1, mu2, var2):
    """Mean and variance of ``|mu1 - mu2|^2`` under the proDA imputation.

    Lifted from ``proDA:::distance_sq``:

        mu     = mu2 - mu1
        sigma  = var1 + var2
        E[d^2] = sum(sigma) + sum(mu^2)
        Var[d^2] = 2 sum(sigma^2) + 4 sum(mu^2 * sigma)
    """
    mu = mu2 - mu1
    sigma = var1 + var2
    valid = np.isfinite(mu) & np.isfinite(sigma)
    if not valid.any():
        return np.nan, np.nan
    mu = mu[valid]
    sigma = sigma[valid]
    mean = float(np.sum(sigma) + np.sum(mu ** 2))
    var = float(2.0 * np.sum(sigma ** 2) + 4.0 * np.sum(mu ** 2 * sigma))
    return mean, var


def _refit_blind(fit):
    """Re-fit the per-protein parameters under an intercept-only design.

    Mirrors ``proDA::dist_approx``'s ``blind=TRUE`` default: distances
    should not be biased by the design used for the original DE test, so
    each protein is re-fit with ``~1`` while the dropout curves stay
    frozen.
    """
    import numpy as np

    from .fit import fit_one_protein

    n_samples = fit.data.shape[1]
    X_blind = np.ones((n_samples, 1))
    rho = fit.hyper_parameters.rho
    zeta = fit.hyper_parameters.zeta
    coefs = np.full((fit.data.shape[0], 1), np.nan)
    cov_list = []
    for i in range(fit.data.shape[0]):
        fi = fit_one_protein(fit.data[i], X_blind, rho=rho, zeta=zeta)
        coefs[i] = fi.coefficients
        cov_list.append(fi.coef_variance_matrix)
    return X_blind, coefs, cov_list


def dist_approx(
    fit,
    *,
    by_sample: bool = False,
    blind: bool = True,
) -> dict:
    r"""Approximate pairwise distance under the proDA dropout model.

    Parameters
    ----------
    fit
        A :class:`pyproda.pipeline.ProDAFit`.
    by_sample
        If ``True`` returns distances between samples; otherwise (default)
        between proteins.
    blind
        If ``True`` (default, matches R), re-fit each protein under an
        intercept-only design so the distances do not depend on the
        design used for differential testing. The dropout curves are
        kept frozen during the blind refit.

    Returns
    -------
    dict
        ``{"mean": pandas.DataFrame, "sd": pandas.DataFrame}``. Each is
        an ``n x n`` symmetric DataFrame (rows = cols = proteins or
        samples). Use ``scipy.spatial.distance.squareform(mean.values,
        checks=False)`` for a condensed-vector form.
    """
    Y = fit.data
    if blind:
        X, coefs, coef_var = _refit_blind(fit)
    else:
        X = fit.design
        coefs = fit.coefficients
        coef_var = fit.coef_variance_matrices
    Pred = coefs @ X.T

    # Imputed values: observed → Y, missing → Pred.
    Mu = np.where(np.isnan(Y), Pred, Y)
    # Pred-variance per (i, j): diag of (X Var(β_i) X').
    n_proteins, n_samples = Y.shape
    Pred_var = np.zeros_like(Y)
    for i in range(n_proteins):
        cov = coef_var[i]
        if cov is None or np.any(~np.isfinite(cov)):
            Pred_var[i] = np.nan
            continue
        Pred_var[i] = np.einsum("ij,jk,ik->i", X, cov, X)
        Pred_var[i] = np.maximum(Pred_var[i], 0.0)

    # Only the *missing* positions carry imputation uncertainty.
    Mu_var = np.where(np.isnan(Y), Pred_var, 0.0)

    if by_sample:
        Mu = Mu.T
        Mu_var = Mu_var.T
        names = fit.sample_names
    else:
        names = fit.protein_names

    n = Mu.shape[0]
    mean_mat = np.zeros((n, n))
    var_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_mean, d_var = _distance_sq(Mu[i], Mu_var[i], Mu[j], Mu_var[j])
            if np.isfinite(d_mean) and d_mean > 0:
                mean_mat[i, j] = np.sqrt(d_mean)
                mean_mat[j, i] = mean_mat[i, j]
                var_mat[i, j] = d_var / (4.0 * d_mean)
                var_mat[j, i] = var_mat[i, j]
            else:
                mean_mat[i, j] = np.nan
                mean_mat[j, i] = np.nan
                var_mat[i, j] = np.nan
                var_mat[j, i] = np.nan

    mean_df = pd.DataFrame(mean_mat, index=names, columns=names)
    sd_df = pd.DataFrame(np.sqrt(np.maximum(var_mat, 0.0)),
                         index=names, columns=names)
    return {"mean": mean_df, "sd": sd_df}
