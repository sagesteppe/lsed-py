import numpy as np
from scipy.special import expit

from lsed import clamp, cox_calibration, recalibrate_to_prevalence


def test_clamp_bounds_and_leaves_interior_untouched():
    assert np.allclose(clamp([0, 1], eps=1e-6), [1e-6, 1 - 1e-6])
    p = np.array([0.2, 0.5, 0.8])
    assert np.allclose(clamp(p), p)
    c = clamp([-5, 5])
    assert np.all((c >= 1e-6) & (c <= 1 - 1e-6))


def test_cox_calibration_returns_the_documented_named_pair():
    rng = np.random.default_rng(20)
    truth = rng.binomial(1, 0.3, size=200)
    prob = expit(rng.normal(size=200))
    cc = cox_calibration(truth, prob)
    assert set(cc) == {"cox_intercept", "cox_slope"}


def test_perfectly_calibrated_large_sample_recovers_slope1_intercept0():
    rng = np.random.default_rng(21)
    n = 20000
    lp = rng.normal(0, 1.5, size=n)        # true linear predictor
    prob = expit(lp)
    truth = rng.binomial(1, prob)          # outcomes drawn from those probs
    cc = cox_calibration(truth, prob)
    assert np.isclose(cc["cox_slope"], 1, atol=0.1)
    assert np.isclose(cc["cox_intercept"], 0, atol=0.1)


def test_separated_fit_is_dropped_to_nan():
    # a cleanly separating predictor on the usual scale makes the slope diverge
    truth = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    cc = cox_calibration(truth, prob)
    assert np.isnan(cc["cox_slope"])


def test_recalibration_is_identity_when_observed_equals_target():
    prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(recalibrate_to_prevalence(prob, p_obs=0.4, pi=0.4), prob,
                       atol=1e-6)


def test_recalibration_preserves_order_and_moves_with_target():
    prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    down = recalibrate_to_prevalence(prob, p_obs=0.4, pi=0.05)
    up = recalibrate_to_prevalence(prob, p_obs=0.4, pi=0.60)
    assert np.array_equal(np.argsort(down), np.argsort(prob))
    assert np.all(down < prob)             # lower target prevalence lowers probs
    assert np.all(up > prob)               # higher target raises them
