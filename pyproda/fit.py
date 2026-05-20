"""
Per-protein censored maximum-likelihood fit.

Mirrors ``proDA::pd_lm.fit`` for the v0.1 sub-set of options
(no ``moderate_location``, no ``moderate_variance``). For each row of
the intensity matrix we fit

    y[s] = X[s,:] @ beta + eps,  eps ~ N(0, sigma^2)

with missing values dropping out via

    P(observed | y) = invprobit((y - rho_s) / zeta_s),

i.e. the joint likelihood over observed and missing positions is

    L = prod_{s in obs} N(y_s; X_s beta, sigma^2)
      * prod_{s in mis} Phi((X_s beta - rho_s) / zeta_s*)

where ``zeta_s* = zeta_s * sqrt(1 + sigma^2 / zeta_s^2)`` is the
marginalisation correction (matches R ``zetastar``).

Fast paths
----------
* All-observed rows: closed-form OLS — ``beta = (X'X)^-1 X' y``,
  ``sigma2 = SSE / n``  (R uses the MLE form ``* (n-p)/n``).
* Otherwise: ``scipy.optimize.minimize`` with BFGS on
  ``(beta, sigma2)``, gradient supplied. Hessian via finite-difference
  on the gradient at the optimum (matches R's ``optim(..., hessian=T)``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .invprobit import inv_mills_ratio, invprobit


@dataclass
class PerProteinFit:
    """Per-protein result for one row of the matrix."""

    coefficients: np.ndarray              # (p,)
    coef_variance_matrix: np.ndarray      # (p, p)
    s2: float                              # residual variance estimate
    df: float                              # degrees of freedom (n_obs - rank(X))
    n_obs: int                             # observed-sample count
    converged: bool = True


def _neg_log_lik(par, X, yo, Xo, Xm, rho, zeta):
    """Negative joint log-likelihood; ``par = [beta, sigma2]``."""
    p = X.shape[1]
    beta = par[:p]
    sigma2 = par[p]
    if sigma2 <= 0:
        return 1e6
    val = 0.0
    if Xo.shape[0] > 0:
        resid = yo - Xo @ beta
        val += stats.norm.logpdf(resid, loc=0.0, scale=np.sqrt(sigma2)).sum()
    if Xm.shape[0] > 0:
        zetastar = zeta * np.sqrt(1.0 + sigma2 / zeta ** 2)
        # P(missing | Xm beta) — invprobit with ``oneminus=True`` (we
        # pass positive zeta; R passes negative zeta and oneminus=False,
        # which is the same thing).
        val += invprobit(Xm @ beta, rho, zetastar, log=True, oneminus=True).sum()
    return -val


def _neg_grad(par, X, yo, Xo, Xm, rho, zeta):
    """Gradient of the joint negative log-likelihood w.r.t.
    ``[beta, sigma2]``. Matches R ``grad_fnc``."""
    p = X.shape[1]
    beta = par[:p]
    sigma2 = par[p]
    if sigma2 <= 0:
        return np.zeros_like(par)

    # Observed-block gradients.
    if Xo.shape[0] > 0:
        dbeta_o = -(Xo.T @ Xo @ beta - Xo.T @ yo) / sigma2
        dsig2_o = float(np.sum(((Xo @ beta - yo) ** 2 - sigma2) / (2 * sigma2 ** 2)))
    else:
        dbeta_o = np.zeros(p)
        dsig2_o = 0.0

    # Missing-block gradients. We model P(missing | y) = 1 - Phi(z), so
    # the inverse Mills ratio uses the ``oneminus=True`` branch (with
    # positive zeta — the Python convention).
    if Xm.shape[0] > 0:
        zetastar = zeta * np.sqrt(1.0 + sigma2 / zeta ** 2)
        imr = inv_mills_ratio(Xm @ beta, rho, zetastar, oneminus=True)
        dbeta_m = Xm.T @ imr
        dsig2_m = -float(np.sum((Xm @ beta - rho) / (2 * zetastar ** 2) * imr))
    else:
        dbeta_m = np.zeros(p)
        dsig2_m = 0.0

    grad = np.concatenate([dbeta_o + dbeta_m, [dsig2_o + dsig2_m]])
    return -grad


def _numeric_hessian(par, X, yo, Xo, Xm, rho, zeta, *, eps: float = 1e-5):
    """Finite-difference Hessian of the negative log-lik."""
    n = par.size
    H = np.zeros((n, n))
    g0 = _neg_grad(par, X, yo, Xo, Xm, rho, zeta)
    for j in range(n):
        e = np.zeros(n)
        h_j = eps * (abs(par[j]) + 1e-3)
        e[j] = h_j
        gj = _neg_grad(par + e, X, yo, Xo, Xm, rho, zeta)
        H[:, j] = (gj - g0) / h_j
    # Symmetrise.
    H = 0.5 * (H + H.T)
    return H


def _invert_psd(mat, *, ridge: float = 0.0) -> np.ndarray:
    """Pseudo-inverse of a (nearly) positive-semi-definite matrix."""
    if ridge:
        mat = mat + ridge * np.eye(mat.shape[0])
    try:
        return np.linalg.pinv(mat)
    except np.linalg.LinAlgError:
        return np.full_like(mat, np.nan)


def fit_one_protein(
    y: np.ndarray,
    X: np.ndarray,
    *,
    rho: Optional[np.ndarray] = None,
    zeta: Optional[np.ndarray] = None,
) -> PerProteinFit:
    r"""Fit the censored linear model for a single protein row.

    Parameters
    ----------
    y
        Length-``n`` vector of intensities; ``NaN`` = missing.
    X
        ``(n, p)`` design matrix.
    rho
        Per-sample dropout positions (length ``n``). ``NaN``-entries are
        allowed for fully-observed samples (they are never indexed).
        If ``None`` (one-shot init, no dropout model yet), the missing
        block has no likelihood contribution and the fit falls back to
        OLS on observed rows.
    zeta
        Per-sample dropout scales (length ``n``). Same NA convention.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    obs_mask = ~np.isnan(y)
    n_obs = int(obs_mask.sum())
    yo = y[obs_mask]
    Xo = X[obs_mask]
    Xm = X[~obs_mask]

    rank_X = int(np.linalg.matrix_rank(X))

    failed = PerProteinFit(
        coefficients=np.full(p, np.nan),
        coef_variance_matrix=np.full((p, p), np.nan),
        s2=np.nan,
        df=np.nan,
        n_obs=n_obs,
        converged=False,
    )

    if n_obs < p:
        # Underdetermined if we have no missing-block likelihood help.
        if rho is None or zeta is None or Xm.shape[0] == 0:
            return failed

    # --- Initial guesses ----------------------------------------------
    if n_obs == 0:
        return failed
    if _has_intercept(X):
        beta_init = np.zeros(p)
        beta_init[0] = float(np.mean(yo))
    else:
        beta_init = np.full(p, float(np.mean(yo)))
    sigma2_init = max(1e-3, float(np.var(yo)) if yo.size > 1 else 1.0)

    if rho is None or zeta is None or Xm.shape[0] == 0:
        # ---- All-observed fast path ---- #
        if n_obs <= rank_X:
            return failed
        XtX_inv = np.linalg.pinv(Xo.T @ Xo)
        beta = XtX_inv @ Xo.T @ yo
        resid = yo - Xo @ beta
        # R uses ``summary(lm)$sigma^2 * (n - p)/n`` (the MLE form,
        # i.e. ``SSE / n``). proDA's downstream df/s2 expects this.
        sigma2 = float((resid ** 2).sum() / n_obs)
        coef_hessian = (Xo.T @ Xo) / max(sigma2, 1e-100)
        Var_coef = _invert_psd(coef_hessian)
        # Unbiased s² for the t-test would be SSE/(n-p), but proDA's
        # ``calculate_sigma2_parameters`` returns ``s2_approx`` based on
        # the MLE; we follow the same path here.
        sigma2_var = 2.0 * sigma2 ** 2 / max(n_obs, 1)
        n_approx, df_approx, s2_approx = _sigma2_params(
            sigma2, sigma2_var, n_obs, rank_X,
        )
        # Recompute the (X' X / s2_approx) Hessian for the *unbiased* var.
        coef_hessian_unbiased = (Xo.T @ Xo) / max(s2_approx, 1e-100)
        Var_coef_unbiased = _invert_psd(coef_hessian_unbiased)
        return PerProteinFit(
            coefficients=beta,
            coef_variance_matrix=Var_coef_unbiased,
            s2=s2_approx,
            df=df_approx,
            n_obs=n_obs,
            converged=True,
        )

    # ---- Censored MLE path ---- #
    rho = np.asarray(rho, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    rho_m = rho[~obs_mask]
    zeta_m = zeta[~obs_mask]

    # If any rho/zeta is NaN in the missing block (means the column was
    # entirely observed when learning the curve), bail out — the curve
    # can't constrain that row.
    if Xm.shape[0] > 0 and (np.isnan(rho_m).any() or np.isnan(zeta_m).any()):
        # Treat missing samples as truly absent (no censoring info):
        Xm = Xm[:0]
        rho_m = rho_m[:0]
        zeta_m = zeta_m[:0]

    par0 = np.concatenate([beta_init, [sigma2_init]])

    def f(par):
        return _neg_log_lik(par, X, yo, Xo, Xm, rho_m, zeta_m)

    def g(par):
        return _neg_grad(par, X, yo, Xo, Xm, rho_m, zeta_m)

    try:
        res = minimize(
            f, par0, jac=g, method="L-BFGS-B",
            bounds=[(None, None)] * p + [(1e-8, None)],
            options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6},
        )
    except (FloatingPointError, ValueError):
        return failed

    if not res.success and (
        not np.isfinite(res.fun) or np.any(~np.isfinite(res.x))
    ):
        return failed

    beta = res.x[:p]
    sigma2 = max(float(res.x[p]), 1e-100)

    # Hessian at optimum via finite differences on the gradient.
    H = _numeric_hessian(res.x, X, yo, Xo, Xm, rho_m, zeta_m)
    coef_hessian = H[:p, :p]
    sigma2_var_inv = H[p, p]
    if sigma2_var_inv <= 0 or not np.isfinite(sigma2_var_inv):
        return failed
    sigma2_var = 1.0 / sigma2_var_inv

    n_approx, df_approx, s2_approx = _sigma2_params(
        sigma2, sigma2_var, n_obs, rank_X,
    )
    # Recompute coef_hessian at the *unbiased* sigma2 — proDA does
    # this so the inverted hessian carries the unbiased Var(coef).
    # We approximate by scaling: H_coef(sigma2_unb) ≈ H_coef * sigma2/s2_approx
    # (only valid for the observed block; missing-block Hessian depends
    # weakly on sigma2). For v0.1 this is close enough.
    if s2_approx > 0 and np.isfinite(s2_approx):
        coef_hessian = coef_hessian * sigma2 / s2_approx
    if np.any(np.diag(coef_hessian) < 0):
        coef_hessian = np.zeros_like(coef_hessian)
    Var_coef = _invert_psd(coef_hessian)

    coef_should_be_inf = np.diag(Var_coef) > 1e6
    if coef_should_be_inf.any():
        for idx in np.flatnonzero(coef_should_be_inf):
            Var_coef[idx, idx] = np.inf
        beta[coef_should_be_inf] = np.nan
        if coef_should_be_inf.all():
            n_approx = np.nan
            df_approx = np.nan
            s2_approx = np.nan

    return PerProteinFit(
        coefficients=beta,
        coef_variance_matrix=Var_coef,
        s2=s2_approx,
        df=df_approx,
        n_obs=n_obs,
        converged=bool(res.success),
    )


def _has_intercept(X: np.ndarray) -> bool:
    return np.any(np.all(np.isclose(X, 1.0), axis=0))


def _sigma2_params(
    sigma2: float, sigma2_var: float, n_obs: int, p: int,
) -> tuple[float, float, float]:
    r"""Approximate ``(n_approx, df_approx, s2_approx)`` from the MLE.

    Mirrors ``proDA::calculate_sigma2_parameters`` for the unmoderated
    case. The MLE of a Gaussian variance is sigma² ~ Inverse-Chi²(n, σ²)
    so:

        E[σ̂²] = (n / (n-2)) σ²,
        Var[σ̂²] = 2 σ⁴ (n / ((n-2)² (n-4))).

    With ``moderate_variance = FALSE`` proDA simply returns
    ``s2_approx = sigma2 * n_obs / df``, ``df = n_obs - p``,
    ``n_approx = n_obs``.
    """
    df = max(n_obs - p, 1e-3)
    if df <= 0 or not np.isfinite(sigma2):
        return float(n_obs), float(df), float(sigma2)
    # Unbiased: SSE/(n-p) instead of SSE/n. proDA's "s2_approx" is the
    # unbiased variance.
    s2_approx = sigma2 * n_obs / df
    return float(n_obs), float(df), float(s2_approx)
