# pyproda

A **pure-Python port of [Bioconductor proDA](https://bioconductor.org/packages/release/bioc/html/proDA.html)** (Ahlmann-Eltze & Anders 2020) — a probabilistic dropout model for differential abundance analysis of mass-spec proteomics data with MNAR missing values.

- Per-protein censored maximum-likelihood fit (`pd_lm`) reimplemented in NumPy + `scipy.optimize.L-BFGS-B`
- Probit dropout curve `P(observed | y) = invprobit((y - rho)/zeta)` matched bit-exactly to R `proDA::invprobit`
- Per-sample dropout-curve hyperparameter MLE (`rho_s`, `zeta_s`) by Nelder-Mead optim
- Wald test for any user-specified contrast (`test_diff`)
- **No `rpy2`**, no R install — all algorithmic code is reimplemented directly
- Wald `t_statistic` and `pval` agree with R proDA at Pearson r > 0.99 on the synthetic 300x12 benchmark
- AnnData-friendly: accepts `np.ndarray` or `pd.DataFrame` (rows = proteins, columns = samples)

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here.

## Install

```bash
pip install pyproda
```

## Quick-start

```python
import numpy as np
import pandas as pd
from pyproda import proda, test_diff

# M: log2 protein intensity matrix (proteins x samples, NaN = missing)
M = pd.read_csv("proteinGroups_log2.tsv", sep="\t", index_col=0)

# Per-sample group labels (or a (n_samples, p) design matrix).
groups = ["control"] * 3 + ["treatment"] * 3

fit = proda(M, design=groups)

# Wald test: "is the treatment vs control coefficient != 0?"
res = test_diff(fit, np.array([0.0, 1.0]))
res.head()
#    name        pval      adj_pval    diff   t_statistic   se      df    avg_abundance  n_obs
#    prot_0042   3.2e-7    2.1e-4      2.13   8.4           0.25    10    18.91          6
#    ...
```

## Low-level API (mirrors R one-to-one)

```python
from pyproda import (
    invprobit,                # proDA::invprobit
    fit_one_protein,           # proDA::pd_lm.fit
    fit_dropout_curves,        # proDA::dropout_curves
    proda,                     # proDA::proDA
    test_diff,                 # proDA::test_diff
)

p_obs = invprobit(y, rho=18.0, zeta=0.5)                  # probit sigmoid
hp    = fit_dropout_curves(M, Pred)                        # rho_s, zeta_s
fit_i = fit_one_protein(M[0], X, rho=hp.rho, zeta=hp.zeta) # censored MLE
```

The `ProDAFit` container exposes every field the R `proDAFit` S4 object does — `coefficients`, `coef_variance_matrices`, `s2`, `df`, `n_obs`, `hyper_parameters` (with `rho`, `zeta`, `mu0`, `sigma20`), `design`, `convergence`.

## R-parity status

| Step | R counterpart | Match |
|---|---|---|
| `invprobit(x, rho, zeta)` | `proDA::invprobit` | **bit-exact** (atol 1e-12) |
| `inv_mills_ratio(...)` | `proDA:::inv_mills_ratio` | bit-exact |
| Per-protein `coefficients` (censored MLE) | `proDA:::pd_lm.fit` | Pearson r > 0.999 |
| Per-protein `s2`, `df` (un-moderated) | `proDA:::calculate_sigma2_parameters` | within ~5% |
| Per-sample `(rho, zeta)` | `proDA:::dropout_curves` | median |diff| < 0.5 |
| Wald `t_statistic` | `proDA:::run_wald_parameter_test` | **Pearson r > 0.99** |
| Wald `pval` | `proDA::test_diff` | **Pearson r > 0.99** |
| Top-DE protein agreement | `proDA::test_diff` | > 95% |

The remaining gap (relative to bit-exact) is dominated by the **outer EM** for the dropout-curve hyperparameters — R proDA runs up to 20 iterations of `(rho, zeta) <-> coefs` with both `Pred` and `Pred_var` updates; v0.1 runs 3-5 fixed-point iterations and only feeds `Pred_var` from the censored fit. The per-protein coefficient at fixed `(rho, zeta)` agrees with R to ~7 decimal places.

## Benchmark

300 proteins x 12 samples, 12% MNAR-censored, 15% planted DE, 5 outer iterations:

| | Wall time |
|---|---|
| R proDA pipeline (no Rscript startup) | ~3.3 s |
| Python pipeline | ~3.4 s |
| R total (including Rscript startup) | ~9.1 s |
| Python total (no JVM/R startup) | ~3.4 s |

`scipy.optimize.L-BFGS-B` is well-matched to R's C-compiled `nlminb` per protein — neither has a clear edge on raw inner-loop speed. The end-to-end win is the absence of Rscript boot time (~5-6 s on the CMAP env).

Run it yourself:

```bash
python examples/benchmark.py --runs 2 --n-proteins 300 --n-per-group 6
```

## Reproducing R results exactly

`tests/r_reference_driver.R` runs the R proDA pipeline (`proDA(...) -> test_diff(...)`) on the same TSV input the Python side dumps. `tests/test_r_parity.py` (skipped if no CMAP / proDA env) checks every numerical field — see the R-parity table above.

```bash
# Run the R-parity tests
pytest tests/test_r_parity.py -v
```

## What's NOT included (yet)

| R function / feature | Status |
|---|---|
| `invprobit` / `inv_mills_ratio` | OK v0.1 |
| Per-protein censored MLE (`pd_lm.fit`, un-moderated) | OK v0.1 |
| Per-sample dropout curves (`dropout_curves`) | OK v0.1 |
| `proDA()` outer EM (3-5 iterations) | OK v0.1 |
| Wald `test_diff(fit, contrast)` | OK v0.1 |
| `moderate_variance` (empirical-Bayes shrinkage on sigma^2) | planned (v0.2) |
| `moderate_location` (Cauchy prior on beta) | planned (v0.2) |
| LRT / F-test (`test_diff(fit, reduced_model=...)`) | planned (v0.2) |
| `predict(fit, type='feature_parameters')` | planned (v0.2) |
| `dist_approx(fit)` | planned (v0.2) |
| `n_subsample` / chunked fitting for large matrices | planned (v0.2) |
| `pd_row_t_test` convenience two-group wrapper | planned (v0.2) |
| `generate_synthetic_data` | planned (v0.3) |

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: `omicverse.protein.tl.de(adata, method='proda')`
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite the original proDA paper:

> Ahlmann-Eltze C., Anders S. **proDA: Probabilistic Dropout Analysis for Identifying Differentially Abundant Proteins in Label-Free Quantitative Mass Spectrometry.** *bioRxiv* (2020). DOI: 10.1101/661496

...and acknowledge omicverse / this repo for the Python port.

## License

GPL-3 — matches the upstream Bioconductor package.
