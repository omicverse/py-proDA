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

See the README for the R-parity table and benchmark.
"""
from __future__ import annotations

from .fit import PerProteinFit, fit_one_protein
from .hyperparams import HyperParameters, fit_dropout_curves
from .invprobit import inv_mills_ratio, invprobit
from .pipeline import ProDAFit, proda
from .test_diff import test_diff

__version__ = "0.1.0"

__all__ = [
    "proda",
    "test_diff",
    "invprobit",
    "inv_mills_ratio",
    "ProDAFit",
    "PerProteinFit",
    "HyperParameters",
    "fit_dropout_curves",
    "fit_one_protein",
]
