# lsed (Python)

Prevalence-corrected evaluation of species-distribution model predictions — a
Python mirror of the [`lsed` R package](https://github.com/sagesteppe/lsed).

## Install

Install the latest version straight from GitHub (needs `git`; no PyPI release yet):

```bash
pip install "git+https://github.com/sagesteppe/lsed-py.git"
```

Pin a specific version with `@`, e.g. `...lsed-py.git@v0.1.0` (tag) or `@main`.

For local development, from a clone of the repo:

```bash
pip install -e .            # editable install
pip install -e ".[test]"    # with the test dependencies
```

## Usage

```python
import numpy as np
import lsed

rng = np.random.default_rng(1)
truth = rng.binomial(1, 0.3, size=500)
prob = 1 / (1 + np.exp(-(-0.8 + 1.5 * truth + rng.normal(size=500))))

lsed.roc_auc(truth, prob)                      # prevalence-invariant ROC-AUC
lsed.ap_pointwise(truth, prob, pi=0.05)        # average precision at pi = 0.05
lsed.nb_exact(truth, prob, pi=0.05, pt=[0.05, 0.1, 0.2])
lsed.cox_calibration(truth, prob)              # {'cox_intercept': ..., 'cox_slope': ...}
```

## What's here

| Function | Purpose |
|---|---|
| `reweight()` | Case weights that re-price an evaluation sample to a target prevalence |
| `roc_auc()` | Prevalence-invariant ROC-AUC via the rank (Mann-Whitney) identity |
| `ap_pointwise()`, `pr_curve_exact()` | Exact average precision / PR curve at a target prevalence |
| `ap_binormal()`, `pr_curve_binormal()`, `ap_scalar_to_prev()` | Equal-variance binormal (summary-metric) route |
| `nb_exact()`, `nb_binormal()`, `nb_treat_all()` | Decision-curve net benefit (per-point, best-achievable, treat-all) |
| `cox_calibration()`, `recalibrate_to_prevalence()`, `clamp()` | Calibration helpers |

Vector-valued helpers return `numpy.ndarray`; the PR-curve helpers return a
`pandas.DataFrame` with `recall` and `precision` columns.

## Relationship to the R package

This is a faithful port of the [`lsed` R package](https://github.com/sagesteppe/lsed), with the same function
names and behaviour — outputs match the R implementation to numerical tolerance.
The R package is the canonical source: see its
[documentation site](https://sagesteppe.github.io/lsed/) and the accompanying
manuscript for the methods, decision-curve background, and worked examples. This
package deliberately keeps that material in one place rather than duplicating it.

## Tests

```bash
pytest
```
