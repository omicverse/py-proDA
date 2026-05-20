"""High-level :func:`proda` wrapper — mirrors ``proDA::proDA``.

The flow is:

1. mean-impute the matrix and run per-protein OLS to bootstrap
   per-sample predictions,
2. fit per-sample dropout curves ``(rho_s, zeta_s)`` against those
   predictions,
3. re-fit each protein's coefficients via censored MLE using the
   learned curves,
4. (optional) outer iterations to refine the dropout curves
   from the censored-MLE predictions; if ``moderate_location`` /
   ``moderate_variance`` is on, the location & variance priors are
   refit in each iteration too,
5. (optional) ``n_subsample`` chunked fitting: fit hyper-parameters on
   a random subset of ``n_subsample`` proteins, then evaluate the
   remaining ones using the frozen curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .fit import fit_one_protein
from .hyperparams import HyperParameters, fit_dropout_curves, initial_predictions
from .invprobit import invprobit


@dataclass
class ProDAFit:
    """Result of :func:`proda`. Mirrors a ``proDAFit`` S4 object."""

    data: np.ndarray                                 # (n_proteins, n_samples) log2
    design: np.ndarray                               # (n_samples, p)
    coef_names: list[str]
    protein_names: list[str]
    sample_names: list[str]
    coefficients: np.ndarray                         # (n_proteins, p)
    coef_variance_matrices: list[np.ndarray]         # list of (p,p)
    s2: np.ndarray                                   # (n_proteins,)
    df: np.ndarray                                   # (n_proteins,)
    n_obs: np.ndarray                                # (n_proteins,)
    hyper_parameters: HyperParameters                # rho/zeta per sample
    convergence: dict = field(default_factory=dict)
    reference_level: Optional[str] = None
    # New v0.1.1 fields — empirical-Bayes moderation parameters.
    df_prior: float = np.nan
    s2_prior: np.ndarray | float = np.nan            # per-protein or scalar
    n_approx: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    location_prior_mean: float = np.nan
    location_prior_scale: float = np.nan
    location_prior_df: float = 3.0
    variance_prior_scale: float = np.nan
    variance_prior_df: float = np.nan
    design_columns: Optional[list[str]] = None
    _design_df: Optional[pd.DataFrame] = None

    # ---- Convenience accessors mirroring the R S4 slots ---- #
    @property
    def dropout_curve_position(self) -> np.ndarray:
        return self.hyper_parameters.rho

    @property
    def dropout_curve_scale(self) -> np.ndarray:
        return self.hyper_parameters.zeta

    # ---- R-style functional accessors (also exposed as module-level) ---- #
    def abundances_(self) -> pd.DataFrame:
        """``proDA::abundances(fit)`` — the original input matrix as a DataFrame.

        R returns the (un-imputed) intensity matrix stored in the
        ``"abundances"`` assay. Missing values stay ``NaN``.
        """
        return pd.DataFrame(
            self.data,
            index=self.protein_names,
            columns=self.sample_names,
        )

    def coefficients_(self) -> pd.DataFrame:
        """``proDA::coefficients(fit)`` — (n_proteins x p) DataFrame."""
        return pd.DataFrame(
            self.coefficients,
            index=self.protein_names,
            columns=self.coef_names,
        )

    def coefficient_variance_matrices_(self) -> list[np.ndarray]:
        """``proDA::coefficient_variance_matrices(fit)``.

        Returns one (p x p) covariance per protein, in protein order.
        """
        return list(self.coef_variance_matrices)

    def convergence_(self) -> dict:
        """``proDA::convergence(fit)`` — outer EM convergence info."""
        return dict(self.convergence)

    def design_(self) -> pd.DataFrame:
        """``proDA::design(fit)`` — the model matrix as a DataFrame."""
        return pd.DataFrame(
            self.design,
            index=self.sample_names,
            columns=self.coef_names,
        )

    def feature_parameters_(self) -> pd.DataFrame:
        """``proDA::feature_parameters(fit)``.

        Per-protein DataFrame with columns ``n_approx``, ``df``, ``s2``,
        ``n_obs`` — matches R column names exactly.
        """
        n_approx = self.n_approx if self.n_approx.size else self.n_obs.astype(float)
        return pd.DataFrame(
            {
                "n_approx": n_approx,
                "df": self.df,
                "s2": self.s2,
                "n_obs": self.n_obs,
            },
            index=self.protein_names,
        )

    def hyper_parameters_(self) -> dict:
        """``proDA::hyper_parameters(fit)``.

        Returns a dict with the global location / variance priors and
        the per-sample dropout curves.
        """
        return {
            "location_prior_mean": float(self.location_prior_mean),
            "location_prior_scale": float(self.location_prior_scale),
            "location_prior_df": float(self.location_prior_df),
            "variance_prior_scale": float(self.variance_prior_scale),
            "variance_prior_df": float(self.variance_prior_df),
            "dropout_curve_position": np.asarray(
                self.hyper_parameters.rho, dtype=float
            ),
            "dropout_curve_scale": np.asarray(
                self.hyper_parameters.zeta, dtype=float
            ),
        }

    def reference_level_(self) -> Optional[str]:
        """``proDA::reference_level(fit)``."""
        return self.reference_level

    def result_names_(self) -> list[str]:
        """``proDA::result_names(fit)`` — names of coefficients/contrasts."""
        return list(self.coef_names)

    # ---- predict() ---- #
    def predict(self, type: str = "response") -> pd.DataFrame:
        r"""``proDA::predict(fit, type="response")``.

        Returns the per-protein/sample predicted log-intensity
        ``coefs @ X.T`` as a DataFrame. ``type="response"`` and
        ``type="link"`` both return the linear-predictor matrix (proDA's
        ``predict`` method is the linear-predictor; the dropout
        observation probability is a separate quantity that callers can
        compute via :func:`invprobit`).
        """
        Pred = self.coefficients @ self.design.T
        df_out = pd.DataFrame(
            Pred,
            index=self.protein_names,
            columns=self.sample_names,
        )
        if type == "response" or type == "link":
            return df_out
        if type == "feature_parameters":
            return self  # type: ignore[return-value]
        raise ValueError(
            f"unknown type={type!r}; expected 'response', 'link', or "
            "'feature_parameters'."
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ProDAFit(n_proteins={len(self.protein_names)}, "
            f"n_samples={len(self.sample_names)}, "
            f"coefs={self.coef_names!r})"
        )


# ---- Module-level R-style functional accessors ---- #
def abundances(fit: ProDAFit) -> pd.DataFrame:
    """``proDA::abundances(fit)`` — see :meth:`ProDAFit.abundances_`."""
    return fit.abundances_()


def coefficients(fit: ProDAFit) -> pd.DataFrame:
    """``proDA::coefficients(fit)`` — see :meth:`ProDAFit.coefficients_`."""
    return fit.coefficients_()


def coefficient_variance_matrices(fit: ProDAFit) -> list[np.ndarray]:
    """``proDA::coefficient_variance_matrices(fit)``."""
    return fit.coefficient_variance_matrices_()


def convergence(fit: ProDAFit) -> dict:
    """``proDA::convergence(fit)`` — outer EM convergence info."""
    return fit.convergence_()


def design(fit: ProDAFit) -> pd.DataFrame:
    """``proDA::design(fit)`` — the model matrix as a DataFrame."""
    return fit.design_()


def feature_parameters(fit: ProDAFit) -> pd.DataFrame:
    """``proDA::feature_parameters(fit)``."""
    return fit.feature_parameters_()


def hyper_parameters(fit: ProDAFit) -> dict:
    """``proDA::hyper_parameters(fit)``."""
    return fit.hyper_parameters_()


def reference_level(fit: ProDAFit) -> Optional[str]:
    """``proDA::reference_level(fit)``."""
    return fit.reference_level_()


def result_names(fit: ProDAFit) -> list[str]:
    """``proDA::result_names(fit)`` — names of coefficients/contrasts."""
    return fit.result_names_()


def predict(fit: ProDAFit, type: str = "response") -> pd.DataFrame:
    """``proDA::predict(fit, type="response")``."""
    return fit.predict(type=type)


def _make_design(
    design,
    M: pd.DataFrame | np.ndarray,
    *,
    reference_level: Optional[str] = None,
) -> tuple[np.ndarray, list[str]]:
    """Resolve ``design`` argument to a (n_samples, p) matrix + names.

    Accepts a numpy array, a DataFrame, or a vector of group labels
    (one per sample) — the latter is one-hot encoded with the optional
    ``reference_level`` dropped (intercept + dummies, R-style).
    """
    n_samples = M.shape[1] if isinstance(M, (np.ndarray, pd.DataFrame)) else len(M)

    if isinstance(design, pd.DataFrame):
        return design.to_numpy(dtype=float), list(design.columns.astype(str))
    if isinstance(design, np.ndarray) and design.ndim == 2:
        return design.astype(float), [f"X{j}" for j in range(design.shape[1])]
    # Treat as a vector of group labels.
    arr = np.asarray(design)
    if arr.ndim == 1 and arr.size == n_samples:
        levels = list(pd.unique(arr))
        if reference_level is not None and reference_level in levels:
            levels.remove(reference_level)
            levels = [reference_level] + levels
        # Intercept + dummies.
        X = np.zeros((n_samples, len(levels)))
        X[:, 0] = 1.0
        for j, lv in enumerate(levels[1:], start=1):
            X[:, j] = (arr == lv).astype(float)
        names = ["Intercept"] + [f"{lv}" for lv in levels[1:]]
        return X, names
    raise ValueError(
        f"could not interpret design of type {type(design).__name__}"
    )


# ---- Empirical-Bayes moderation (Smyth-style) ---- #
def _trigamma_inverse(x: float, *, max_iter: int = 50,
                     atol: float = 1e-8) -> float:
    """Newton solve for ``y`` such that ``trigamma(y) = x``.

    Lifted from ``pydeqms.ebayes._trigamma_inverse`` (which itself
    matches ``limma::trigammaInverse``). Used to estimate the prior
    degrees of freedom ``d0`` in :func:`_estimate_variance_prior`.
    """
    from scipy import special

    if x <= 0:
        return float("inf")
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(max_iter):
        tg = special.polygamma(1, y)
        d = tg * (1 - tg / x) / special.polygamma(2, y)
        y += d
        if abs(d / y) < atol:
            break
    return float(y)


def _estimate_variance_prior(
    s2: np.ndarray, df: np.ndarray,
) -> tuple[float, float]:
    r"""Empirical-Bayes prior ``(s2_prior, df_prior)``.

    Faithful port of ``proDA:::variance_prior`` — the per-protein
    ``s2`` are assumed inverse-chi-square distributed, so ``s2 / tau²``
    follows an ``F(df, df0)`` distribution. We estimate ``(tau², df0)``
    by maximising the F log-likelihood:

        argmax_{tau, dfinv}  sum  log f_F(s2/tau ; df1=df, df2=1/dfinv)
                                  - log(tau)

    via Nelder-Mead, exactly as the R code does. This is robust to the
    pathological ``s2`` outliers that R's ``calculate_sigma2_parameters``
    produces for under-determined rows (which break the Smyth
    moment estimator).

    Returns ``(s2_prior, df_prior) = (tau20, 1/df0_inv)``.
    """
    from scipy import optimize, stats

    good = (
        np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    )
    if good.sum() < 2:
        return float("nan"), float("nan")
    s2g = s2[good]
    dfg = df[good]

    def neg_ll(par):
        tau, dfinv = par
        if tau <= 0 or dfinv <= 0:
            return np.inf
        df2 = 1.0 / dfinv
        with np.errstate(divide="ignore", invalid="ignore"):
            logf = stats.f.logpdf(s2g / tau, dfn=dfg, dfd=df2) - np.log(tau)
        val = np.nansum(logf[np.isfinite(logf)])
        if not np.isfinite(val):
            return np.inf
        return -val

    res = optimize.minimize(
        neg_ll, x0=np.array([1.0, 1.0]), method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8},
    )
    tau20 = float(res.x[0])
    df0_inv = float(res.x[1])
    if df0_inv <= 0 or not np.isfinite(df0_inv):
        return float("nan"), float("nan")
    df_prior = 1.0 / df0_inv
    return float(tau20), float(df_prior)


def _estimate_location_prior(
    Pred: np.ndarray, Pred_var: np.ndarray, trim: float = 0.2,
) -> tuple[float, float]:
    """Trimmed-mean location prior ``(mu0, sigma20)`` for the β prior.

    Mirrors ``proDA:::location_prior`` for the un-blinded case:
    ``mu0 = trim-mean(Pred, 0.2)``, ``sigma20`` from the upper half
    via the simple moment estimator
    ``sigma20 = mean((pred - mu0)^2 - pred_var)``.
    """
    from scipy import stats

    finite = np.isfinite(Pred)
    if not finite.any():
        return float("nan"), float("nan")
    pred_flat = Pred[finite]
    mu0 = float(stats.trim_mean(pred_flat, proportiontocut=trim))
    # Upper-half (away from dropout regime) for the scale.
    if Pred_var is None:
        Pred_var = np.zeros_like(Pred)
    upper = (Pred > mu0) & finite
    if not upper.any():
        return mu0, float(np.var(pred_flat, ddof=1))
    pred_u = (Pred[upper] - mu0).ravel()
    pvar_u = np.asarray(Pred_var)[upper].ravel()
    finite_u = np.isfinite(pred_u) & np.isfinite(pvar_u)
    if not finite_u.any():
        return mu0, float(np.var(pred_flat, ddof=1))
    pred_u = pred_u[finite_u]
    pvar_u = pvar_u[finite_u]
    sigma20 = float(np.mean(np.maximum(pred_u ** 2 - pvar_u, 0.0)))
    if sigma20 <= 0 or not np.isfinite(sigma20):
        sigma20 = float(np.var(pred_flat, ddof=1))
    return mu0, sigma20


def proda(
    M: np.ndarray | pd.DataFrame,
    design,
    *,
    reference_level: Optional[str] = None,
    data_is_log_transformed: bool = True,
    moderate_variance: bool = False,
    moderate_location: bool = False,
    location_prior_df: float = 3.0,
    n_subsample: Optional[int] = None,
    max_iter: int = 3,
    epsilon: float = 1e-3,
    verbose: bool = False,
    seed: int = 0,
) -> ProDAFit:
    r"""Probabilistic-dropout-model fit.

    Parameters
    ----------
    M
        Protein × sample (log-)intensity matrix. NaN = missing.
    design
        ``(n_samples, p)`` design matrix, OR a length-``n_samples``
        vector of group labels (one-hot encoded internally).
    reference_level
        If ``design`` is a label vector, which label is the reference
        (gets dropped from the dummy encoding). Default: first label.
    data_is_log_transformed
        If ``False``, the matrix is ``log2``-transformed before fitting
        (matches R default behaviour).
    moderate_variance
        If ``True``, apply empirical-Bayes shrinkage to per-protein
        ``s2`` (Smyth 2004 moment-matching). The posterior variance
        ``s2_post = (df_prior*s2_prior + df*s2) / (df_prior + df)`` is
        used in the downstream Wald test.
    moderate_location
        If ``True``, add a location prior on β (trimmed-mean estimator
        of ``mu0, sigma20``); see ``proDA:::location_prior``.
    location_prior_df
        Cauchy-style df for the location prior (default 3, matches R).
    n_subsample
        If set and ``< n_proteins``, fit hyper-parameters on a random
        ``n_subsample`` rows only; the remaining proteins re-use the
        frozen curves for per-row censored MLE (chunked fitting).
    max_iter
        Number of outer (rho/zeta ↔ coefs) iterations. R default is 20;
        v0.1 converged in 3-5; moderate_* might need a couple more.
    epsilon
        Convergence tolerance on (rho, zeta) between iterations.
    verbose
        Print per-iteration status.
    seed
        RNG seed for the initial mean-impute step.
    """
    # ---- Parse / coerce inputs ----
    if isinstance(M, pd.DataFrame):
        protein_names = list(M.index.astype(str))
        sample_names = list(M.columns.astype(str))
        M_arr = M.to_numpy(dtype=float)
    else:
        M_arr = np.asarray(M, dtype=float)
        protein_names = [f"prot_{i:04d}" for i in range(M_arr.shape[0])]
        sample_names = [f"S{j:02d}" for j in range(M_arr.shape[1])]

    # Treat exact zeros as missing (matches R warning + replacement
    # when the matrix has no NAs).
    if not np.isnan(M_arr).any() and np.any(M_arr == 0):
        M_arr = np.where(M_arr == 0, np.nan, M_arr)

    if not data_is_log_transformed:
        with np.errstate(divide="ignore", invalid="ignore"):
            M_arr = np.log2(M_arr)

    X, coef_names = _make_design(design, M_arr, reference_level=reference_level)
    if X.shape[0] != M_arr.shape[1]:
        raise ValueError(
            f"design has {X.shape[0]} rows but M has {M_arr.shape[1]} "
            "columns; design rows must match M samples."
        )

    n_proteins, n_samples = M_arr.shape

    # n_subsample chunked fitting.
    if n_subsample is None or n_subsample >= n_proteins:
        sub_idx = np.arange(n_proteins)
        rest_idx = np.array([], dtype=int)
    else:
        rng = np.random.default_rng(seed)
        sub_idx = np.sort(rng.choice(n_proteins, size=n_subsample, replace=False))
        rest_idx = np.setdiff1d(np.arange(n_proteins), sub_idx)

    M_sub = M_arr[sub_idx]

    # ---- Iter 0: mean-imputed OLS → predictions ----
    coefs, Pred = initial_predictions(M_sub, X, seed=seed)
    Pred_var = np.zeros_like(Pred)
    # Initial per-protein OLS s2 (for moderation init).
    s2_init = np.full(M_sub.shape[0], np.nan)
    df_init = np.full(M_sub.shape[0], np.nan)
    for i in range(M_sub.shape[0]):
        y = M_sub[i]
        obs = ~np.isnan(y)
        if obs.sum() > X.shape[1]:
            resid = y[obs] - X[obs] @ coefs[i]
            s2_init[i] = float((resid ** 2).sum() / max(obs.sum() - X.shape[1], 1))
            df_init[i] = obs.sum() - X.shape[1]

    # Initial moderation parameters.
    if moderate_variance:
        s2_prior_scalar, df_prior_scalar = _estimate_variance_prior(s2_init, df_init)
    else:
        s2_prior_scalar, df_prior_scalar = float("nan"), float("nan")
    if moderate_location:
        mu0, sigma20 = _estimate_location_prior(Pred, Pred_var)
    else:
        mu0, sigma20 = float("nan"), float("nan")

    last_rho = None
    last_zeta = None
    cov_list: list[np.ndarray] = []
    s2_vec = np.full(M_sub.shape[0], np.nan)
    df_vec = np.full(M_sub.shape[0], np.nan)
    n_approx_vec = np.full(M_sub.shape[0], np.nan)
    n_obs_vec = np.zeros(M_sub.shape[0], dtype=int)
    converged = False
    error = float("nan")
    it = 0
    for it in range(1, max_iter + 1):
        # Per-sample dropout-curve MLE.
        hp = fit_dropout_curves(M_sub, Pred, Pred_var)

        # Per-protein censored MLE with the new curves.
        coefs_new = np.full((M_sub.shape[0], X.shape[1]), np.nan)
        cov_list = []
        s2_vec = np.full(M_sub.shape[0], np.nan)
        df_vec = np.full(M_sub.shape[0], np.nan)
        n_approx_vec = np.full(M_sub.shape[0], np.nan)
        n_obs_vec = np.zeros(M_sub.shape[0], dtype=int)
        Pred_var = np.zeros_like(Pred)
        for i in range(M_sub.shape[0]):
            fit_i = fit_one_protein(
                M_sub[i], X,
                rho=hp.rho, zeta=hp.zeta,
            )
            coefs_new[i] = fit_i.coefficients
            cov_list.append(fit_i.coef_variance_matrix)
            s2_vec[i] = fit_i.s2
            df_vec[i] = fit_i.df
            n_approx_vec[i] = fit_i.n_obs  # placeholder until moderation refines
            n_obs_vec[i] = fit_i.n_obs
            if cov_list[-1] is not None and np.all(np.isfinite(cov_list[-1])):
                XCX = (X @ cov_list[-1] * X).sum(axis=1)
                Pred_var[i] = np.maximum(XCX, 0.0)
        coefs = np.where(np.isnan(coefs_new), coefs, coefs_new)
        Pred = coefs @ X.T

        # Re-estimate priors (moderation) for next iteration.
        if moderate_variance:
            s2_prior_scalar, df_prior_scalar = _estimate_variance_prior(
                s2_vec, df_vec,
            )
        if moderate_location:
            mu0, sigma20 = _estimate_location_prior(Pred, Pred_var)

        if verbose:
            print(
                f"[proda] iter {it}: rho≈{np.nanmedian(hp.rho):.3f}, "
                f"zeta≈{np.nanmedian(hp.zeta):.3f}, "
                f"s2_prior={s2_prior_scalar:.3f}, df_prior={df_prior_scalar:.2f}"
            )

        if last_rho is not None and last_zeta is not None:
            d_rho = float(np.nanmean((hp.rho - last_rho) ** 2))
            d_zeta = float(np.nanmean((hp.zeta - last_zeta) ** 2))
            error = d_rho + d_zeta
            if d_rho < epsilon and d_zeta < epsilon:
                converged = True
                break
        last_rho = hp.rho.copy()
        last_zeta = hp.zeta.copy()
    else:
        # max_iter reached without break.
        converged = False

    # ---- Apply EB shrinkage to s2 (posterior) ---- #
    if moderate_variance and np.isfinite(df_prior_scalar) and np.isfinite(s2_prior_scalar):
        s2_post = np.where(
            np.isfinite(s2_vec) & np.isfinite(df_vec) & (df_vec > 0),
            (df_prior_scalar * s2_prior_scalar + df_vec * s2_vec)
            / (df_prior_scalar + df_vec),
            s2_vec,
        )
        df_total = df_vec + df_prior_scalar
        # Scale the coefficient variance matrices by s2_post / s2.
        new_cov: list[np.ndarray] = []
        for i, cov in enumerate(cov_list):
            if (
                cov is not None
                and np.all(np.isfinite(cov))
                and np.isfinite(s2_vec[i])
                and s2_vec[i] > 0
            ):
                scale = s2_post[i] / s2_vec[i]
                new_cov.append(cov * scale)
            else:
                new_cov.append(cov)
        cov_list = new_cov
        s2_vec = s2_post
        df_vec = df_total

    # ---- Now fit the remaining proteins (n_subsample chunking) ---- #
    if rest_idx.size > 0:
        coefs_rest = np.full((rest_idx.size, X.shape[1]), np.nan)
        cov_rest: list[np.ndarray] = []
        s2_rest = np.full(rest_idx.size, np.nan)
        df_rest = np.full(rest_idx.size, np.nan)
        n_approx_rest = np.full(rest_idx.size, np.nan)
        n_obs_rest = np.zeros(rest_idx.size, dtype=int)
        for k, gi in enumerate(rest_idx):
            fit_i = fit_one_protein(
                M_arr[gi], X,
                rho=hp.rho, zeta=hp.zeta,
            )
            coefs_rest[k] = fit_i.coefficients
            cov_rest.append(fit_i.coef_variance_matrix)
            s2_rest[k] = fit_i.s2
            df_rest[k] = fit_i.df
            n_approx_rest[k] = fit_i.n_obs
            n_obs_rest[k] = fit_i.n_obs
        if moderate_variance and np.isfinite(df_prior_scalar) and np.isfinite(s2_prior_scalar):
            s2_post_rest = np.where(
                np.isfinite(s2_rest) & np.isfinite(df_rest) & (df_rest > 0),
                (df_prior_scalar * s2_prior_scalar + df_rest * s2_rest)
                / (df_prior_scalar + df_rest),
                s2_rest,
            )
            df_total_rest = df_rest + df_prior_scalar
            new_cov_rest = []
            for i, cov in enumerate(cov_rest):
                if (
                    cov is not None
                    and np.all(np.isfinite(cov))
                    and np.isfinite(s2_rest[i])
                    and s2_rest[i] > 0
                ):
                    scale = s2_post_rest[i] / s2_rest[i]
                    new_cov_rest.append(cov * scale)
                else:
                    new_cov_rest.append(cov)
            cov_rest = new_cov_rest
            s2_rest = s2_post_rest
            df_rest = df_total_rest

        # Stitch sub + rest back into protein order.
        all_coefs = np.full((n_proteins, X.shape[1]), np.nan)
        all_coefs[sub_idx] = coefs
        all_coefs[rest_idx] = coefs_rest
        all_cov: list[np.ndarray] = [None] * n_proteins  # type: ignore[list-item]
        for k, gi in enumerate(sub_idx):
            all_cov[gi] = cov_list[k]
        for k, gi in enumerate(rest_idx):
            all_cov[gi] = cov_rest[k]
        all_s2 = np.full(n_proteins, np.nan)
        all_s2[sub_idx] = s2_vec
        all_s2[rest_idx] = s2_rest
        all_df = np.full(n_proteins, np.nan)
        all_df[sub_idx] = df_vec
        all_df[rest_idx] = df_rest
        all_n_obs = np.zeros(n_proteins, dtype=int)
        all_n_obs[sub_idx] = n_obs_vec
        all_n_obs[rest_idx] = n_obs_rest
        all_n_approx = np.full(n_proteins, np.nan)
        all_n_approx[sub_idx] = n_approx_vec
        all_n_approx[rest_idx] = n_approx_rest

        coefs = all_coefs
        cov_list = all_cov
        s2_vec = all_s2
        df_vec = all_df
        n_obs_vec = all_n_obs
        n_approx_vec = all_n_approx

    convergence_info = {
        "iterations": it,
        "successful": bool(converged),
        "error": float(error) if np.isfinite(error) else None,
    }

    return ProDAFit(
        data=M_arr,
        design=X,
        coef_names=coef_names,
        protein_names=protein_names,
        sample_names=sample_names,
        coefficients=coefs,
        coef_variance_matrices=cov_list,
        s2=s2_vec,
        df=df_vec,
        n_obs=n_obs_vec,
        hyper_parameters=hp,
        convergence=convergence_info,
        reference_level=reference_level,
        df_prior=float(df_prior_scalar),
        s2_prior=float(s2_prior_scalar),
        n_approx=n_approx_vec,
        location_prior_mean=float(mu0),
        location_prior_scale=float(sigma20),
        location_prior_df=float(location_prior_df),
        variance_prior_scale=float(s2_prior_scalar),
        variance_prior_df=float(df_prior_scalar),
    )
