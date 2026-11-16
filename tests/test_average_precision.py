import numpy as np
from scipy.stats import norm

from lsed import (
    ap_binormal,
    ap_pointwise,
    ap_scalar_to_prev,
    pr_curve_binormal,
    pr_curve_exact,
)


def test_ap_pointwise_nan_without_both_classes_and_bounded():
    assert np.isnan(ap_pointwise(np.ones(5), np.random.rand(5), pi=0.1))
    assert np.isnan(ap_pointwise(np.zeros(5), np.random.rand(5), pi=0.1))
    rng = np.random.default_rng(30)
    truth = rng.binomial(1, 0.3, size=200)
    ap = ap_pointwise(truth, rng.random(200), pi=0.1)
    assert 0 <= ap <= 1


def test_perfect_ranking_gives_ap_one_at_any_prevalence():
    truth = np.array([0, 0, 0, 1, 1])
    prob = np.array([1, 2, 3, 4, 5])       # the two presences score highest
    assert np.isclose(ap_pointwise(truth, prob, pi=0.5), 1)
    assert np.isclose(ap_pointwise(truth, prob, pi=0.01), 1)


def test_ap_binormal_equals_prevalence_baseline_with_no_discrimination():
    assert np.isclose(ap_binormal(0, 0.1), 0.1, atol=1e-3)
    assert np.isclose(ap_binormal(0, 0.4), 0.4, atol=1e-3)


def test_ap_binormal_rises_with_separation_and_stays_within_pi_1():
    pi = 0.1
    a0, a1, a2 = ap_binormal(0.5, pi), ap_binormal(1.5, pi), ap_binormal(3, pi)
    assert a0 < a1 < a2
    assert pi < a2 < 1


def test_ap_scalar_to_prev_inverts_ap_binormal_and_is_self_consistent():
    pr_auc, pi0, pi_target = 0.4, 0.1, 0.03
    res = ap_scalar_to_prev(pr_auc, pi0, pi_target)
    assert set(res) == {"ap_target", "mu", "implied_roc_auc"}
    # the fitted mu reproduces the reported PR-AUC at the design prevalence
    assert np.isclose(ap_binormal(res["mu"], pi0, grid=20000), pr_auc, atol=1e-4)
    # ap_target is the same ROC re-integrated at the new prevalence
    assert np.isclose(res["ap_target"],
                      ap_binormal(res["mu"], pi_target, grid=20000))
    # implied ROC-AUC follows the binormal relation norm.cdf(mu / sqrt(2))
    assert np.isclose(res["implied_roc_auc"], norm.cdf(res["mu"] / np.sqrt(2)))


def test_ap_pointwise_and_ap_binormal_agree_on_data_from_the_roc():
    # large binormal sample: the point-level estimator should track the
    # closed-form binormal AP once re-priced to the same prevalence.
    rng = np.random.default_rng(31)
    mu, pi, n = 1.5, 0.1, 200_000
    truth = rng.binomial(1, 0.5, size=n)             # sample prevalence ~ 0.5
    prob = rng.normal(loc=mu * truth, scale=1.0)     # binormal scores
    assert np.isclose(ap_pointwise(truth, prob, pi), ap_binormal(mu, pi),
                      atol=0.02)


def test_pr_curve_exact_shape_and_monotone_recall():
    rng = np.random.default_rng(32)
    truth = rng.binomial(1, 0.3, size=150)
    prob = 1 / (1 + np.exp(-(-0.5 + truth + rng.normal(size=150))))
    cv = pr_curve_exact(truth, prob, pi=0.1)
    assert list(cv.columns) == ["recall", "precision"]
    assert len(cv) == len(truth)
    assert np.all(np.diff(cv["recall"].to_numpy()) >= 0)   # recall never decreases
    assert np.all(np.isfinite(cv["precision"].to_numpy()))


def test_pr_curve_binormal_grid_length_and_valid_precisions():
    cv = pr_curve_binormal(mu=1.2, pi=0.1, grid=250)
    assert list(cv.columns) == ["recall", "precision"]
    assert len(cv) == 250
    assert np.all((cv["precision"] > 0) & (cv["precision"] < 1))
    assert np.all((cv["recall"] > 0) & (cv["recall"] < 1))
