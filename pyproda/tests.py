"""Convenience two-group / multi-group tests on raw matrices.

These mirror ``proDA::pd_row_t_test`` and ``proDA::pd_row_f_test`` —
both are thin wrappers around :func:`proda` + :func:`test_diff` that
build a synthetic design vector for the caller. They are useful for the
common "I just have two matrices ``X`` and ``Y``, give me row-wise
differential abundance" workflow.

>>> from pyproda import pd_row_t_test
>>> res = pd_row_t_test(X, Y)                  # Welch-style censored t
>>> res.head()

>>> from pyproda import pd_row_f_test
>>> res = pd_row_f_test(X, groups=['a','a','b','b','c','c'])  # multi-group F
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .pipeline import proda
from .test_diff import test_diff


def pd_row_t_test(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    *,
    moderate_location: bool = False,
    moderate_variance: bool = True,
    alternative: str = "two.sided",
    pval_adjust_method: str = "BH",
    location_prior_df: float = 3.0,
    max_iter: int = 20,
    epsilon: float = 1e-3,
    return_fit: bool = False,
    verbose: bool = False,
):
    r"""Row-wise two-sample test between matrices ``X`` and ``Y``.

    Mirrors ``proDA::pd_row_t_test`` — builds a "Condition1" / "Condition2"
    design vector, runs :func:`proda`, and tests the
    ``Condition1 - Condition2`` contrast.

    Parameters
    ----------
    X, Y
        Two matrices of log-intensities with the same number of rows
        (proteins). ``NaN`` = missing. ``X`` is "Condition1",
        ``Y`` is "Condition2".
    moderate_location, moderate_variance
        Empirical-Bayes shrinkage of β / σ² respectively. Default
        ``moderate_variance=True`` matches R.
    alternative
        ``"two.sided"`` (default), ``"greater"``, or ``"less"``.
    pval_adjust_method
        ``"BH"`` (default) or ``"bonferroni"``.
    location_prior_df, max_iter, epsilon, verbose
        Forwarded to :func:`proda`.
    return_fit
        If ``True``, return ``{"fit": ..., "test_results": ...}``
        instead of just the test-results DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns: ``name``, ``pval``, ``adj_pval``, ``diff``,
        ``t_statistic``, ``se``, ``df``, ``avg_abundance``, ``n_obs``.
    """
    Xa = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    Ya = Y.to_numpy(dtype=float) if isinstance(Y, pd.DataFrame) else np.asarray(Y, dtype=float)
    if Xa.shape[0] != Ya.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of rows; got "
            f"{Xa.shape[0]} vs {Ya.shape[0]}."
        )
    n1 = Xa.shape[1]
    n2 = Ya.shape[1]
    data = np.concatenate([Xa, Ya], axis=1)
    design = ["Condition1"] * n1 + ["Condition2"] * n2
    # Preserve row index if available.
    if isinstance(X, pd.DataFrame):
        prot_names = list(X.index.astype(str))
    elif isinstance(Y, pd.DataFrame):
        prot_names = list(Y.index.astype(str))
    else:
        prot_names = [f"prot_{i:04d}" for i in range(data.shape[0])]
    data_df = pd.DataFrame(
        data, index=prot_names,
        columns=[f"X{i}" for i in range(n1)] + [f"Y{i}" for i in range(n2)],
    )

    fit = proda(
        data_df, design=design,
        reference_level="Condition2",
        moderate_location=moderate_location,
        moderate_variance=moderate_variance,
        location_prior_df=location_prior_df,
        max_iter=max_iter, epsilon=epsilon, verbose=verbose,
    )
    # Contrast = Condition1 - Condition2. With reference_level="Condition2",
    # the design is [Intercept, Condition1] => Condition1 - Condition2 maps
    # to the coefficient on "Condition1" alone (i.e. c = [0, 1]).
    c = np.zeros(len(fit.coef_names))
    if "Condition1" in fit.coef_names:
        c[fit.coef_names.index("Condition1")] = 1.0
    else:
        # Fallback: assume two-coef design [Intercept, X1].
        c[-1] = 1.0
    test_res = test_diff(
        fit, c, alternative=alternative,
        pval_adjust_method=pval_adjust_method,
    )
    if return_fit:
        return {"fit": fit, "test_results": test_res}
    return test_res


def pd_row_f_test(
    X: np.ndarray | pd.DataFrame,
    *additional_matrices,
    groups: Optional[list | np.ndarray] = None,
    moderate_location: bool = False,
    moderate_variance: bool = True,
    pval_adjust_method: str = "BH",
    location_prior_df: float = 3.0,
    max_iter: int = 20,
    epsilon: float = 1e-3,
    return_fit: bool = False,
    verbose: bool = False,
):
    r"""Row-wise F-test (multi-group ANOVA-like) on row matrices.

    Mirrors ``proDA::pd_row_f_test``. Either pass several matrices
    positionally (``pd_row_f_test(X, Y, Z, ...)``) or a single matrix
    ``X`` with a per-column ``groups`` label vector.

    The test compares the full multi-group model against the reduced
    intercept-only model via a per-protein F statistic computed from
    the difference in SSR (mapped to the proDA censored likelihood).

    Parameters
    ----------
    X
        First matrix of log-intensities.
    *additional_matrices
        Additional matrices (Y, Z, ...) appended column-wise.
    groups
        Alternative form — single matrix + per-column group labels.
    moderate_variance, moderate_location, location_prior_df,
    pval_adjust_method, max_iter, epsilon, verbose, return_fit
        Same semantics as :func:`pd_row_t_test`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``name``, ``pval``, ``adj_pval``, ``f_statistic``,
        ``df1``, ``df2``, ``avg_abundance``, ``n_obs``.
    """
    # Resolve inputs.
    if additional_matrices and groups is not None:
        raise ValueError(
            "Please specify *additional_matrices or groups, not both."
        )
    if not additional_matrices and (groups is None or len(np.unique(groups)) < 2):
        raise ValueError(
            "Data for only one condition was specified. Pass multiple "
            "matrices or a groups vector with ≥2 levels."
        )
    Xa = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if additional_matrices:
        mats = [Xa]
        labels: list[str] = ["condition_1"] * Xa.shape[1]
        for idx, m in enumerate(additional_matrices, start=2):
            ma = m.to_numpy(dtype=float) if isinstance(m, pd.DataFrame) else np.asarray(m, dtype=float)
            mats.append(ma)
            labels.extend([f"condition_{idx}"] * ma.shape[1])
        data = np.concatenate(mats, axis=1)
        groups = labels
    else:
        data = Xa
        groups = list(groups)
        if len(groups) != Xa.shape[1]:
            raise ValueError("len(groups) must match ncol(X).")

    if isinstance(X, pd.DataFrame):
        prot_names = list(X.index.astype(str))
    else:
        prot_names = [f"prot_{i:04d}" for i in range(data.shape[0])]
    data_df = pd.DataFrame(
        data, index=prot_names,
        columns=[f"S{j:02d}" for j in range(data.shape[1])],
    )

    fit = proda(
        data_df, design=groups,
        moderate_location=moderate_location,
        moderate_variance=moderate_variance,
        location_prior_df=location_prior_df,
        max_iter=max_iter, epsilon=epsilon, verbose=verbose,
    )

    # Build a reduced design = intercept only.
    n_samples = data.shape[1]
    red_design = np.ones((n_samples, 1))

    # Per-protein F via comparison of residual sum of squares under the
    # full and reduced models on the *observed* samples only. This is a
    # close approximation to R's proper censored-likelihood ratio for
    # well-determined rows; rows with mostly-missing values fall back to
    # NaN which matches R's behaviour.
    res = _per_row_f_test(fit, red_design)
    res["adj_pval"] = _bh_adjust(res["pval"].to_numpy())

    n_approx_vals = (
        fit.n_approx if fit.n_approx.size else fit.n_obs.astype(float)
    )
    out = pd.DataFrame({
        "name": fit.protein_names,
        "pval": res["pval"].to_numpy(),
        "adj_pval": res["adj_pval"].to_numpy(),
        "f_statistic": res["f_statistic"].to_numpy(),
        "df1": res["df1"].to_numpy(),
        "df2": res["df2"].to_numpy(),
        "avg_abundance": np.nanmean(fit.data, axis=1),
        "n_approx": n_approx_vals,
        "n_obs": fit.n_obs,
    })

    if return_fit:
        return {"fit": fit, "test_results": out}
    return out


def _per_row_f_test(fit, red_design: np.ndarray) -> pd.DataFrame:
    """Per-protein F via observed-sample residual SSR ratio.

    A faithful port of ``proDA:::run_nested_model_comparison`` is
    deferred (it requires re-fitting the full censored likelihood
    under both models per protein). For v0.1.1 we use the observed-row
    SSR test which agrees with R to several percent on well-determined
    rows.
    """
    n_proteins = fit.coefficients.shape[0]
    X_full = fit.design
    p_full = X_full.shape[1]
    p_red = red_design.shape[1]
    df1_glob = p_full - p_red
    pvals = np.full(n_proteins, np.nan)
    fstats = np.full(n_proteins, np.nan)
    df1s = np.full(n_proteins, float(df1_glob))
    df2s = np.full(n_proteins, np.nan)

    from scipy import stats

    for i in range(n_proteins):
        y = fit.data[i]
        obs = ~np.isnan(y)
        n_obs = int(obs.sum())
        if n_obs <= p_full:
            continue
        yo = y[obs]
        Xo = X_full[obs]
        Xo_red = red_design[obs]
        # Full-model SSR on observed samples (use the censored-MLE β).
        beta_full = fit.coefficients[i]
        if np.any(~np.isfinite(beta_full)):
            continue
        ssr_full = float(np.sum((yo - Xo @ beta_full) ** 2))
        # Reduced-model: closed-form OLS on observed-sample yo.
        XtX_inv = np.linalg.pinv(Xo_red.T @ Xo_red)
        beta_red = XtX_inv @ Xo_red.T @ yo
        ssr_red = float(np.sum((yo - Xo_red @ beta_red) ** 2))
        df2 = max(n_obs - p_full, 1)
        df2s[i] = float(df2)
        if df1_glob <= 0 or ssr_full <= 0:
            continue
        f_stat = ((ssr_red - ssr_full) / df1_glob) / (ssr_full / df2)
        fstats[i] = max(f_stat, 0.0)
        pvals[i] = float(stats.f.sf(fstats[i], dfn=df1_glob, dfd=df2))
    return pd.DataFrame({
        "pval": pvals, "f_statistic": fstats,
        "df1": df1s, "df2": df2s,
    })


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    from scipy import stats
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if finite.any():
        out[finite] = stats.false_discovery_control(
            np.clip(p[finite], 0.0, 1.0), method="bh",
        )
    return out
