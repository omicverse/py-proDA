"""
pyproda: Pure-Python port of Bioconductor proDA.

Probabilistic-dropout-model differential abundance for mass-spec
proteomics (Ahlmann-Eltze & Anders 2020). Each protein is fit by a
censored MLE assuming missing values follow a probit dropout:

    P(observed | y) = invprobit((y - rho_s) / zeta_s)

Per-sample ``(rho_s, zeta_s)`` are learned jointly with the per-protein
linear-model coefficients. Differential abundance is then tested via
the standard Wald statistic on a user-specified contrast.

Quick-start
-----------
>>> from pyproda import proda, test_diff
>>> # M: protein × sample log2 intensities (NaN = missing)
>>> # design: vector of group labels OR a (n_samples, p) matrix
>>> fit = proda(M, design=["A","A","A","B","B","B"])
>>> # Test "is the B vs A difference different from 0?"
>>> import numpy as np
>>> res = test_diff(fit, np.array([0.0, 1.0]))  # contrast on coefs

R-functional accessors are also exposed (``coefficients(fit)``,
``feature_parameters(fit)``, ``hyper_parameters(fit)``, ``predict(fit)``,
...) plus convenience two-group / multi-group tests and the
``dist_approx`` probabilistic distance.

See the README for the R-parity table and benchmark.
"""
from __future__ import annotations

from .distance import dist_approx
from .fit import PerProteinFit, fit_one_protein
from .hyperparams import HyperParameters, fit_dropout_curves
from .invprobit import inv_mills_ratio, invprobit
from .normalization import median_normalization
from .pipeline import (
    ProDAFit,
    abundances,
    coefficient_variance_matrices,
    coefficients,
    convergence,
    design,
    feature_parameters,
    hyper_parameters,
    predict,
    proda,
    reference_level,
    result_names,
)
from .test_diff import test_diff
from .tests import pd_row_f_test, pd_row_t_test

__version__ = "0.1.1"

__all__ = [
    # Core pipeline.
    "proda",
    "test_diff",
    "ProDAFit",
    # Low-level building blocks (R one-to-one).
    "invprobit",
    "inv_mills_ratio",
    "PerProteinFit",
    "HyperParameters",
    "fit_dropout_curves",
    "fit_one_protein",
    # R-style functional accessors.
    "abundances",
    "coefficients",
    "coefficient_variance_matrices",
    "convergence",
    "design",
    "feature_parameters",
    "hyper_parameters",
    "predict",
    "reference_level",
    "result_names",
    # Convenience two/multi-group tests.
    "pd_row_t_test",
    "pd_row_f_test",
    # Probabilistic distance.
    "dist_approx",
    # Normalization.
    "median_normalization",
]
