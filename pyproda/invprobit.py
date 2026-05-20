"""
Port of ``proDA::invprobit`` — the inverse-probit dropout sigmoid.

The probability that an intensity ``y`` is *observed* (not missing /
NA) at sample ``s`` is modelled as

    P(observed | y) = invprobit(y, rho_s, zeta_s)
                    = Phi((y - rho_s) / zeta_s)              if zeta_s > 0
                    = 1 - Phi((y - rho_s) / (-zeta_s))       if zeta_s < 0

where ``Phi`` is the standard normal CDF, ``rho_s`` is the dropout
*position* (intensity at which P(observed)=0.5) and ``zeta_s`` is the
dropout *scale* (width of the sigmoid; negative values flip the curve).

This matches the R reference exactly:

>>> R: pnorm(x, mean = rho, sd = zeta)                 # zeta > 0
>>> R: pnorm(x, mean = rho, sd = -zeta, lower.tail = TRUE, oneminus=TRUE)  # zeta < 0
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def invprobit(
    x,
    rho,
    zeta,
    *,
    log: bool = False,
    oneminus: bool = False,
) -> np.ndarray:
    r"""Probit dropout curve: ``P(observed | y=x) = Phi((x - rho)/zeta)``.

    Parameters
    ----------
    x
        Intensity value(s). Scalar or array-like.
    rho
        Dropout position (50% observation threshold). Same broadcasting
        rules as ``x``.
    zeta
        Dropout scale (sigmoid width). Sign-sensitive: positive ``zeta``
        gives a curve where high intensities are observed (the proteomics
        default); negative ``zeta`` flips it (used internally during
        per-sample dropout-curve estimation where the parameter
        ``zetainv = -1/sqrt(sigma20)``).
    log
        If ``True`` return ``log P(observed)``.
    oneminus
        If ``True`` return ``1 - P(observed) = P(missing)`` (or its log
        when ``log=True``).

    Returns
    -------
    numpy.ndarray
        Same broadcast shape as ``x``, ``rho``, ``zeta``.
    """
    x = np.asarray(x, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    zeta_arr = np.asarray(zeta, dtype=float)

    # All zeta entries must share a sign (matches R guard); pick sign
    # from the first finite entry.
    finite_zeta = zeta_arr[np.isfinite(zeta_arr)]
    if finite_zeta.size > 0:
        sign_sum = float(np.sign(finite_zeta).sum())
        if abs(sign_sum) < finite_zeta.size:
            raise ValueError("All zeta values must have the same sign.")
        sign_z = -1.0 if sign_sum < 0 else 1.0
    else:
        sign_z = 1.0

    if sign_z < 0:
        # R: pnorm(x, mean=rho, sd=-zeta, lower.tail=oneminus, log.p=log)
        sd = -zeta_arr
        lower_tail = oneminus
    else:
        # R: pnorm(x, mean=rho, sd=zeta, lower.tail=!oneminus, log.p=log)
        sd = zeta_arr
        lower_tail = not oneminus

    z = (x - rho_arr) / sd
    if log:
        if lower_tail:
            out = stats.norm.logcdf(z)
        else:
            out = stats.norm.logsf(z)
    else:
        if lower_tail:
            out = stats.norm.cdf(z)
        else:
            out = stats.norm.sf(z)
    return np.asarray(out, dtype=float)


def inv_mills_ratio(x, mu, sd, *, oneminus: bool = False) -> np.ndarray:
    r"""Inverse Mills ratio.

    With ``oneminus=False`` (default): ``phi(z) / Phi(sign(sd) * z)`` —
    matches ``proDA:::inv_mills_ratio``. Used by the gradient of the
    censored-MLE objective when we have observed-block invprobit.

    With ``oneminus=True``: derivative of ``log(1 - invprobit(...))``
    instead — needed when we model ``P(missing | y) = 1 - Phi(z)`` (the
    Python convention with positive ``zeta``).
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd_arr = np.asarray(sd, dtype=float)
    sign_sd = np.sign(sd_arr)
    abs_sd = np.abs(sd_arr)
    log_dnorm = stats.norm.logpdf(x, loc=mu, scale=abs_sd)
    log_invp = invprobit(x, mu, sd_arr, log=True, oneminus=oneminus)
    factor = -sign_sd if oneminus else sign_sd
    return factor * np.exp(log_dnorm - log_invp)
