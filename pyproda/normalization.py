"""Median normalization — port of ``proDA::median_normalization``.

For each sample (column) we subtract the median of
``X[:, j] - rowMeans(X)`` from that column, so the row-wise means
are aligned across samples. ``NaN`` entries are ignored.

>>> from pyproda import median_normalization
>>> Mn, offsets = median_normalization(M)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def median_normalization(
    X: np.ndarray | pd.DataFrame,
    spike_in_rows: Optional[np.ndarray | list] = None,
) -> Tuple[np.ndarray | pd.DataFrame, np.ndarray]:
    r"""Per-sample median centering.

    Parameters
    ----------
    X
        ``(n_proteins, n_samples)`` matrix (or DataFrame). ``NaN`` =
        missing.
    spike_in_rows
        Optional integer or boolean index of "spike-in" rows used to
        compute the per-sample offset (default: all rows).

    Returns
    -------
    Xnorm
        Same type as ``X`` (DataFrame in / out, ndarray in / out).
    offsets
        Length-``n_samples`` array of per-sample shifts that were
        subtracted from each column.
    """
    is_df = isinstance(X, pd.DataFrame)
    Xa = X.to_numpy(dtype=float) if is_df else np.asarray(X, dtype=float)
    n_proteins, n_samples = Xa.shape

    if spike_in_rows is None:
        rows = np.arange(n_proteins)
    else:
        arr = np.asarray(spike_in_rows)
        if arr.dtype == bool:
            if arr.size != n_proteins:
                raise ValueError(
                    "spike_in_rows boolean vector must have one entry per row."
                )
            if not arr.any():
                raise ValueError(
                    "All entries of spike_in_rows are FALSE; need ≥1 TRUE."
                )
            rows = np.flatnonzero(arr)
        else:
            rows = arr

    sub = Xa[rows, :]
    row_means = np.nanmean(sub, axis=1)  # length n_spike_rows
    offsets = np.empty(n_samples, dtype=float)
    for j in range(n_samples):
        delta = sub[:, j] - row_means
        offsets[j] = float(np.nanmedian(delta))
    Xnorm = Xa - offsets[None, :]

    if is_df:
        return pd.DataFrame(Xnorm, index=X.index, columns=X.columns), offsets
    return Xnorm, offsets
