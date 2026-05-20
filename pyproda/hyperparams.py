"""
Per-sample dropout-curve hyper-parameters (``rho_s``, ``zeta_s``).

Mirrors ``proDA:::dropout_curves`` for the v0.1 unmoderated path. For
each *sample* (column) we maximise

    L(rho_s, zeta_s) =
        N(rho_s | mu0, sigma20)
      + log|zetainv_s|                              (zetainv = -1/zeta)
      + sum_{i in obs(s)}  log [ 1 - Phi((y_is - rho_s)/zeta_s) ]
      + sum_{i in miss(s)} log Phi((Pred_is - rho_s) / zetastar_is)

where ``Pred`` is the current per-protein/sample prediction
(``coefs @ X.T``) and ``mu0, sigma20`` are the 20%-trimmed mean and
variance of ``Pred``.

For v0.1 we use a single one-pass EM:

* initialise ``Pred`` by per-row mean-imputation of NAs + OLS,
* fit each sample's ``(rho_s, zeta_s)`` with ``optim`` (BFGS),
* re-fit each protein's censored MLE using the learned curves.

This is enough to produce a usable ``test_diff`` and matches R
``proDA(...)`` to within a few percent on synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .invprobit import invprobit


@dataclass
class HyperParameters:
    rho: np.ndarray            # (n_samples,)
    zeta: np.ndarray           # (n_samples,)   == 1 / zetainv
    zetainv: np.ndarray        # (n_samples,)   = -1 / zeta  (R convention)
    mu0: float
    sigma20: float


def _trimmed_mean(x: np.ndarray, trim: float = 0.2) -> float:
    return float(stats.trim_mean(x[~np.isnan(x)], proportiontocut=trim))


def _trimmed_var(x: np.ndarray, mu0: float, trim: float = 0.2) -> float:
    sq = (x - mu0) ** 2
    sq = sq[~np.isnan(sq)]
    return float(stats.trim_mean(sq, proportiontocut=trim))


def fit_dropout_curves(
    Y: np.ndarray,
    Pred: np.ndarray,
    Pred_var: Optional[np.ndarray] = None,
) -> HyperParameters:
    r"""Per-sample dropout-curve MLE.

    Parameters
    ----------
    Y
        ``(n_proteins, n_samples)`` intensity matrix (NaN = missing).
    Pred
        ``(n_proteins, n_samples)`` current model predictions
        ``coefs @ X.T``.
    Pred_var
        Optional variance of the predictions at each (i, s) entry — used
        to inflate ``zetastar`` for missing positions. If ``None`` it's
        treated as zero (still a reasonable v0.1).
    """
    Y = np.asarray(Y, dtype=float)
    Pred = np.asarray(Pred, dtype=float)
    n_proteins, n_samples = Y.shape
    if Pred_var is None:
        Pred_var = np.zeros_like(Pred)

    finite_pred = Pred[np.isfinite(Pred)]
    if finite_pred.size == 0:
        finite_pred = Y[np.isfinite(Y)]
    mu0 = _trimmed_mean(finite_pred)
    sigma20 = _trimmed_var(finite_pred, mu0)
    if sigma20 <= 0 or not np.isfinite(sigma20):
        sigma20 = 5.0

    rho = np.full(n_samples, np.nan)
    zetainv = np.full(n_samples, np.nan)

    for s in range(n_samples):
        y_col = Y[:, s]
        is_mis = np.isnan(y_col)
        if not is_mis.any():
            continue
        yo = y_col[~is_mis]
        predm = Pred[is_mis, s]
        pred_var_m = Pred_var[is_mis, s]

        def neg_ll(par):
            rho_s, zinv = par[0], par[1]
            if zinv >= 0:
                return np.inf
            zeta_s = 1.0 / zinv  # negative
            val = stats.norm.logpdf(rho_s, loc=mu0, scale=np.sqrt(sigma20))
            val += min(np.log(abs(zinv)), np.log(10_000.0))
            # P(observed) for the actually observed samples: invprobit
            # with oneminus=TRUE  (R: invprobit(yo, par[1], 1/par[2],
            # log=TRUE, oneminus=TRUE)).
            val += np.nansum(
                invprobit(yo, rho_s, zeta_s, log=True, oneminus=True)
            )
            # P(missing) for samples that *are* missing — invprobit on
            # the predicted intensity, with zetastar inflated by
            # Pred_var (R: sign(par[2]) * sqrt(1/par[2]^2 + pred_var_m)).
            sd_m = np.sign(zinv) * np.sqrt(1.0 / zinv ** 2 + pred_var_m)
            val += np.nansum(invprobit(predm, rho_s, sd_m, log=True))
            return -val

        x0 = np.array([mu0, -1.0 / np.sqrt(sigma20)])
        try:
            res = minimize(
                neg_ll, x0, method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6},
            )
            rho[s] = float(res.x[0])
            zinv_s = float(res.x[1])
            if abs(zinv_s) > 10_000.0:
                zinv_s = -10_000.0
            zetainv[s] = zinv_s
        except Exception:
            rho[s] = mu0
            zetainv[s] = -1.0 / np.sqrt(max(sigma20, 1e-3))

    # R stores zetainv = -1/zeta; positive zeta is what proDA uses
    # downstream (positive-sign sigmoid).
    zeta = np.abs(1.0 / zetainv)
    return HyperParameters(rho=rho, zeta=zeta, zetainv=zetainv,
                           mu0=mu0, sigma20=sigma20)


def initial_predictions(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Per-protein OLS on a mean-imputed copy of ``Y``.

    Used to bootstrap the dropout-curve estimator on the very first
    iteration. Returns ``(coefs, Pred)`` of shapes ``(n_proteins, p)``
    and ``(n_proteins, n_samples)``.
    """
    rng = np.random.default_rng(seed)
    n_proteins, n_samples = Y.shape
    p = X.shape[1]
    Y_compl = Y.copy()
    mask = np.isnan(Y_compl)
    if mask.any():
        # R fills with normal(quantile(Y, 0.1), sd(Y)/5).
        valid = Y[~np.isnan(Y)]
        if valid.size > 0:
            fill_mean = float(np.quantile(valid, 0.1))
            fill_sd = float(np.std(valid, ddof=1)) / 5.0
        else:
            fill_mean, fill_sd = 0.0, 1.0
        Y_compl[mask] = rng.normal(fill_mean, max(fill_sd, 1e-3),
                                    size=int(mask.sum()))

    XtX_inv = np.linalg.pinv(X.T @ X)
    coefs = (XtX_inv @ X.T @ Y_compl.T).T  # (n_proteins, p)
    Pred = coefs @ X.T
    return coefs, Pred
