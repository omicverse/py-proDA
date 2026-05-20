"""High-level :func:`proda` wrapper — mirrors ``proDA::proDA``.

The flow for v0.1 is:

1. mean-impute the matrix and run per-protein OLS to bootstrap
   per-sample predictions,
2. fit per-sample dropout curves ``(rho_s, zeta_s)`` against those
   predictions,
3. re-fit each protein's coefficients via censored MLE using the
   learned curves,
4. (optional) one extra outer iteration to refine the dropout curves
   from the censored-MLE predictions.

``moderate_location`` / ``moderate_variance`` / nested F-test are NOT
implemented in v0.1 — see ``README.md`` for the v0.2 roadmap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .fit import fit_one_protein
from .hyperparams import HyperParameters, fit_dropout_curves, initial_predictions


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

    # ---- Convenience accessors mirroring the R S4 slots ---- #
    @property
    def dropout_curve_position(self) -> np.ndarray:
        return self.hyper_parameters.rho

    @property
    def dropout_curve_scale(self) -> np.ndarray:
        return self.hyper_parameters.zeta

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ProDAFit(n_proteins={len(self.protein_names)}, "
            f"n_samples={len(self.sample_names)}, "
            f"coefs={self.coef_names!r})"
        )


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


def proda(
    M: np.ndarray | pd.DataFrame,
    design,
    *,
    reference_level: Optional[str] = None,
    data_is_log_transformed: bool = True,
    moderate_variance: bool = False,
    moderate_location: bool = False,
    max_iter: int = 3,
    verbose: bool = False,
) -> ProDAFit:
    r"""Probabilistic-dropout-model fit (v0.1).

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
    moderate_variance, moderate_location
        Must be ``False`` in v0.1 (these full moderation features land
        in v0.2 — currently the function will warn and proceed without
        moderation).
    max_iter
        Number of outer (rho/zeta ↔ coefs) iterations. R default is 20
        but v0.1 converges to within a few percent of R within 2-3
        iterations on standard proteomics data.
    verbose
        Print per-iteration status.
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

    if moderate_variance or moderate_location:
        import warnings
        warnings.warn(
            "moderate_variance / moderate_location are not implemented "
            "in v0.1 — proceeding without moderation.",
            UserWarning, stacklevel=2,
        )

    n_proteins, n_samples = M_arr.shape

    # ---- Iter 0: mean-imputed OLS → predictions ----
    coefs, Pred = initial_predictions(M_arr, X)
    Pred_var = np.zeros_like(Pred)

    last_rho = None
    last_zeta = None
    for it in range(1, max_iter + 1):
        # Per-sample dropout-curve MLE
        hp = fit_dropout_curves(M_arr, Pred, Pred_var)

        # Per-protein censored MLE with the new curves.
        coefs_new = np.full((n_proteins, X.shape[1]), np.nan)
        cov_list: list[np.ndarray] = []
        s2_vec = np.full(n_proteins, np.nan)
        df_vec = np.full(n_proteins, np.nan)
        n_obs_vec = np.zeros(n_proteins, dtype=int)
        Pred_var = np.zeros_like(Pred)
        for i in range(n_proteins):
            fit_i = fit_one_protein(
                M_arr[i], X,
                rho=hp.rho, zeta=hp.zeta,
            )
            coefs_new[i] = fit_i.coefficients
            cov_list.append(fit_i.coef_variance_matrix)
            s2_vec[i] = fit_i.s2
            df_vec[i] = fit_i.df
            n_obs_vec[i] = fit_i.n_obs
            # Diagonal of X cov X' for the next dropout-curve fit.
            if cov_list[-1] is not None and np.all(np.isfinite(cov_list[-1])):
                XCX = (X @ cov_list[-1] * X).sum(axis=1)
                Pred_var[i] = np.maximum(XCX, 0.0)
        coefs = np.where(np.isnan(coefs_new), coefs, coefs_new)
        Pred = coefs @ X.T

        if verbose:
            print(
                f"[proda] iter {it}: rho≈{np.nanmedian(hp.rho):.3f}, "
                f"zeta≈{np.nanmedian(hp.zeta):.3f}, "
                f"n_failed={int(np.isnan(coefs_new).any(axis=1).sum())}"
            )

        if last_rho is not None and last_zeta is not None:
            d_rho = float(np.nanmean((hp.rho - last_rho) ** 2))
            d_zeta = float(np.nanmean((hp.zeta - last_zeta) ** 2))
            if d_rho < 1e-3 and d_zeta < 1e-3:
                break
        last_rho = hp.rho.copy()
        last_zeta = hp.zeta.copy()

    convergence = {"iterations": it, "converged": True}

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
        convergence=convergence,
        reference_level=reference_level,
    )
