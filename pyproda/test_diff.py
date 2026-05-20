"""
Wald test for a linear contrast on a fitted ``ProDAFit``.

Mirrors ``proDA::test_diff(fit, contrast=...)`` for the v0.1 path:

* ``contrast`` is a length-``p`` numeric vector (in design-coef space),
* the t-statistic is ``c' beta / sqrt(c' Var(beta) c)``,
* the df is ``n_samples - p`` (matches R ``run_wald_parameter_test``
  for the un-moderated case).

The F-test / nested-model path is *not* implemented in v0.1.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats


def test_diff(fit, contrast, *, alternative: str = "two.sided",
              pval_adjust_method: str = "BH") -> pd.DataFrame:
    r"""Wald test for a per-protein linear contrast.

    Parameters
    ----------
    fit
        A :class:`pyproda.pipeline.ProDAFit` object.
    contrast
        Length-``p`` numeric vector (one entry per design coefficient)
        or a name from ``fit.coef_names`` for the simple "test this
        coefficient = 0" case.
    alternative
        ``"two.sided"`` (default), ``"greater"``, or ``"less"``.
    pval_adjust_method
        Passed to ``scipy.stats.false_discovery_control`` (BH only for
        v0.1).

    Returns
    -------
    pandas.DataFrame
        Columns: ``name``, ``pval``, ``adj_pval``, ``diff``,
        ``t_statistic``, ``se``, ``df``, ``avg_abundance``, ``n_obs``.
    """
    coefs = fit.coefficients
    cov_list = fit.coef_variance_matrices
    n_proteins, p = coefs.shape

    if isinstance(contrast, str):
        if contrast not in fit.coef_names:
            raise KeyError(
                f"contrast {contrast!r} not in design columns {fit.coef_names}"
            )
        c = np.zeros(p)
        c[fit.coef_names.index(contrast)] = 1.0
    else:
        c = np.asarray(contrast, dtype=float)
        if c.shape != (p,):
            raise ValueError(
                f"contrast must have shape ({p},); got {c.shape}"
            )

    diff = coefs @ c
    diff_var = np.array([
        float(c @ cov @ c) if cov is not None and np.all(np.isfinite(cov))
        else np.nan
        for cov in cov_list
    ])
    se = np.sqrt(diff_var)
    diff_df = fit.design.shape[0] - p
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = diff / se
    with np.errstate(invalid="ignore"):
        pval_lower = stats.t.cdf(t_stat, df=diff_df)
    if alternative == "two.sided":
        pval = 2.0 * np.minimum(pval_lower, 1.0 - pval_lower)
    elif alternative == "greater":
        pval = 1.0 - pval_lower
    elif alternative == "less":
        pval = pval_lower
    else:
        raise ValueError(f"unknown alternative={alternative!r}")

    # BH multiple-testing correction. Handle NaN p-values gracefully.
    finite = np.isfinite(pval)
    adj_pval = np.full_like(pval, np.nan)
    if finite.any():
        if pval_adjust_method.upper() in ("BH", "FDR"):
            adj_pval[finite] = stats.false_discovery_control(
                np.clip(pval[finite], 0.0, 1.0), method="bh",
            )
        elif pval_adjust_method.lower() == "bonferroni":
            adj_pval[finite] = np.minimum(pval[finite] * finite.sum(), 1.0)
        else:
            adj_pval[finite] = pval[finite]

    avg_abundance = np.nanmean(fit.data, axis=1)

    return pd.DataFrame({
        "name": fit.protein_names,
        "pval": pval,
        "adj_pval": adj_pval,
        "diff": diff,
        "t_statistic": t_stat,
        "se": se,
        "df": np.full(n_proteins, float(diff_df)),
        "avg_abundance": avg_abundance,
        "n_obs": fit.n_obs,
    })
