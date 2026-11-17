import numpy as np
import pytest

from lsed import (
    ap_binormal,
    ap_scalar_to_prev,
    nb_binormal,
    nb_binormal_band,
    nb_exact,
    nb_exact_ci,
    nb_treat_all,
)


def test_treat_all_closed_form_and_zero_at_pt_equals_pi():
    pi = 0.1
    pt = np.array([0.02, 0.05, 0.1, 0.2])
    assert np.allclose(nb_treat_all(pi, pt), pi - (1 - pi) * (pt / (1 - pt)))
    # treating everyone breaks even exactly when the threshold equals prevalence
    assert np.isclose(nb_treat_all(pi, pi)[0], 0)


def test_nb_exact_is_vectorised_and_bounded_by_prevalence():
    rng = np.random.default_rng(10)
    truth = rng.binomial(1, 0.3, size=300)
    prob = 1 / (1 + np.exp(-(-0.8 + 1.5 * truth + rng.normal(size=300))))
    pt = [0.02, 0.05, 0.1, 0.2, 0.3]
    nb = nb_exact(truth, prob, pi=0.05, pt=pt)
    assert len(nb) == len(pt)
    assert np.all(nb <= 0.05 + 1e-9)       # never exceeds prevalence


def test_nb_exact_collapses_to_treat_all_when_everyone_is_flagged():
    # prob = 1 >= every pt < 1, so all units are 'treated' -> treat-all
    truth = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    pt = [0.02, 0.1, 0.25]
    pi = 0.08
    assert np.allclose(nb_exact(truth, np.ones(truth.size), pi, pt),
                       nb_treat_all(pi, pt))


def test_nb_exact_is_zero_when_nothing_clears_the_threshold():
    truth = np.array([1, 0, 0, 1, 0])
    assert np.allclose(nb_exact(truth, np.zeros(5), pi=0.1, pt=[0.3, 0.5]),
                       [0, 0])


def test_nb_binormal_vectorised_finite_and_improves_with_separation():
    pi = 0.1
    pt = np.array([0.05, 0.1, 0.2])
    nb0 = nb_binormal(0.5, pi, pt)
    nb1 = nb_binormal(2.0, pi, pt)
    assert len(nb0) == len(pt)
    assert np.all(np.isfinite(nb0))
    assert np.all(nb1 >= nb0 - 1e-9)       # more separable ROC never worse


def test_best_achievable_dominates_treat_all():
    pi = 0.1
    pt = np.array([0.05, 0.1, 0.2, 0.3])
    assert np.all(nb_binormal(1.5, pi, pt) >= nb_treat_all(pi, pt) - 1e-9)


def test_nb_exact_ci_returns_lower_band_around_point_estimate():
    rng = np.random.default_rng(1)
    truth = rng.binomial(1, 0.3, size=400)
    prob = 1 / (1 + np.exp(-(-0.8 + 1.5 * truth + rng.normal(size=400))))
    pt = np.array([0.02, 0.05, 0.1, 0.2])
    ci = nb_exact_ci(truth, prob, pi=0.05, pt=pt, B=199, random_state=1)
    assert list(ci["pt"]) == list(pt)
    assert np.allclose(ci["nb"].to_numpy(), nb_exact(truth, prob, 0.05, pt))
    assert np.all(ci["lower"].to_numpy() <= ci["nb"].to_numpy() + 1e-9)
    assert "upper" not in ci.columns          # lower-only default


def test_nb_exact_ci_two_sided_brackets_and_widens():
    rng = np.random.default_rng(2)
    truth = rng.binomial(1, 0.3, size=400)
    prob = 1 / (1 + np.exp(-(-0.8 + 1.5 * truth + rng.normal(size=400))))
    pt = np.array([0.05, 0.1, 0.2])
    ci80 = nb_exact_ci(truth, prob, 0.05, pt, B=299, level=0.8,
                       side="both", random_state=3)
    ci95 = nb_exact_ci(truth, prob, 0.05, pt, B=299, level=0.95,
                       side="both", random_state=3)
    assert np.all(ci80["lower"].to_numpy() <= ci80["nb"].to_numpy() + 1e-9)
    assert np.all(ci80["upper"].to_numpy() >= ci80["nb"].to_numpy() - 1e-9)
    # higher confidence -> wider (or equal) interval
    w80 = ci80["upper"].to_numpy() - ci80["lower"].to_numpy()
    w95 = ci95["upper"].to_numpy() - ci95["lower"].to_numpy()
    assert np.all(w95 >= w80 - 1e-9)


def test_nb_binormal_band_brackets_central_curve_lower_only_by_default():
    pi0, pi = 0.1, 0.05
    pt = np.array([0.02, 0.05, 0.1, 0.2])
    pr = ap_binormal(1.5, pi0, grid=3000)
    b = nb_binormal_band(pr, pi, pt, spread=0.03, pi0=pi0)
    assert np.all(b["lower"].to_numpy() <= b["nb"].to_numpy() + 1e-9)
    assert "upper" not in b.columns           # lower-only default
    bb = nb_binormal_band(pr, pi, pt, spread=0.03, pi0=pi0, side="both")
    assert np.all(bb["upper"].to_numpy() >= bb["nb"].to_numpy() - 1e-9)
    mu = bb.attrs["mu"]
    assert mu["lower"] < mu["central"] < mu["upper"]
    # wider PR-AUC spread -> wider band
    bw = nb_binormal_band(pr, pi, pt, spread=0.06, pi0=pi0, side="both")
    w = bb["upper"].to_numpy() - bb["lower"].to_numpy()
    ww = bw["upper"].to_numpy() - bw["lower"].to_numpy()
    assert np.all(ww >= w - 1e-9)


def test_nb_binormal_band_inverts_pr_auc_through_ap_binormal_at_pi0():
    pi0, pi = 0.1, 0.03
    pt = np.array([0.02, 0.05, 0.1])
    pr = ap_binormal(1.2, pi0, grid=3000)
    b = nb_binormal_band(pr, pi, pt, spread=0.02, pi0=pi0)
    # central mu recovers the generating separation and prices NB at the target pi
    assert np.isclose(b.attrs["mu"]["central"], 1.2, atol=1e-2)
    assert np.allclose(b["nb"].to_numpy(),
                       nb_binormal(b.attrs["mu"]["central"], pi, pt))


def test_nb_exact_ci_point_estimate_and_columns_match_nb_exact():
    rng = np.random.default_rng(11)
    truth = rng.binomial(1, 0.3, size=200)
    prob = 1 / (1 + np.exp(-(-0.8 + 1.5 * truth + rng.normal(size=200))))
    pt = np.array([0.02, 0.05, 0.1, 0.2])
    ci = nb_exact_ci(truth, prob, 0.05, pt, B=199, random_state=11)
    assert np.allclose(ci["pt"].to_numpy(), pt)
    assert np.allclose(ci["nb"].to_numpy(), nb_exact(truth, prob, 0.05, pt))
    assert list(ci.columns) == ["pt", "nb", "lower"]


def test_nb_exact_ci_is_reproducible_given_a_fixed_random_state():
    rng = np.random.default_rng(0)
    truth = rng.binomial(1, 0.3, size=300)
    prob = 1 / (1 + np.exp(-(-0.8 + 1.5 * truth + rng.normal(size=300))))
    pt = np.array([0.05, 0.1, 0.2])
    a = nb_exact_ci(truth, prob, 0.05, pt, B=299, random_state=99)
    b = nb_exact_ci(truth, prob, 0.05, pt, B=299, random_state=99)
    assert a.equals(b)


def test_nb_exact_ci_unstratified_path_runs_and_brackets_the_estimate():
    rng = np.random.default_rng(7)
    truth = rng.binomial(1, 0.3, size=400)
    prob = 1 / (1 + np.exp(-(-0.8 + 1.5 * truth + rng.normal(size=400))))
    pt = np.array([0.05, 0.1, 0.2])
    ci = nb_exact_ci(truth, prob, 0.05, pt, B=299, side="both",
                     stratified=False, random_state=7)
    assert np.allclose(ci["nb"].to_numpy(), nb_exact(truth, prob, 0.05, pt))
    assert np.all(ci["lower"].to_numpy() <= ci["nb"].to_numpy() + 1e-9)
    assert np.all(ci["upper"].to_numpy() >= ci["nb"].to_numpy() - 1e-9)


def test_nb_exact_ci_band_collapses_when_every_unit_is_flagged():
    # prob = 1 >= every pt < 1, so every stratified resample reproduces treat-all
    # exactly (TPR = FPR = 1 regardless of which indices are drawn): zero width.
    truth = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    pt = np.array([0.02, 0.1, 0.25])
    ci = nb_exact_ci(truth, np.ones(truth.size), 0.08, pt, B=199,
                     side="both", random_state=3)
    assert np.allclose(ci["lower"].to_numpy(), ci["nb"].to_numpy())
    assert np.allclose(ci["upper"].to_numpy(), ci["nb"].to_numpy())
    assert np.allclose(ci["nb"].to_numpy(), nb_treat_all(0.08, pt))


def test_nb_exact_ci_rejects_an_unknown_side():
    truth = np.array([1, 0, 1, 0])
    prob = np.array([0.9, 0.1, 0.8, 0.2])
    with pytest.raises(ValueError):
        nb_exact_ci(truth, prob, 0.1, 0.1, B=9, side="upper")


def test_nb_binormal_band_pi0_defaults_to_pi_and_mu_matches_ap_scalar_to_prev():
    pi0, pi = 0.1, 0.03
    pt = np.array([0.02, 0.05, 0.1])
    pr = ap_binormal(1.4, pi0, grid=3000)
    # omitting pi0 prices the PR-AUC at the target prevalence itself
    b_def = nb_binormal_band(pr, pi, pt, spread=0.02)
    b_pi = nb_binormal_band(pr, pi, pt, spread=0.02, pi0=pi)
    assert b_def.equals(b_pi)
    # central mu is exactly the ap_scalar_to_prev reconstruction at pi0
    b = nb_binormal_band(pr, pi, pt, spread=0.02, pi0=pi0)
    assert np.isclose(b.attrs["mu"]["central"],
                      ap_scalar_to_prev(pr, pi0, pi, grid=3000)["mu"], atol=1e-6)


def test_nb_binormal_band_edges_equal_nb_binormal_at_fitted_mu():
    pi0, pi = 0.08, 0.04
    pt = np.array([0.03, 0.07, 0.15])
    pr = ap_binormal(1.6, pi0, grid=3000)
    b = nb_binormal_band(pr, pi, pt, spread=0.025, pi0=pi0, side="both")
    mu = b.attrs["mu"]
    assert np.allclose(b["lower"].to_numpy(), nb_binormal(mu["lower"], pi, pt))
    assert np.allclose(b["upper"].to_numpy(), nb_binormal(mu["upper"], pi, pt))
    assert np.allclose(b["nb"].to_numpy(), nb_binormal(mu["central"], pi, pt))


def test_nb_binormal_band_carries_only_central_and_lower_mu_when_one_sided():
    b = nb_binormal_band(0.8, 0.05, np.array([0.05, 0.1]), spread=0.03, pi0=0.1)
    assert list(b.attrs["mu"].keys()) == ["central", "lower"]
    assert "upper" not in b.attrs["mu"]


def test_nb_binormal_band_accepts_scalar_pt_and_rejects_unknown_side():
    b = nb_binormal_band(0.8, 0.05, 0.1, spread=0.03, pi0=0.1)
    assert len(b) == 1
    with pytest.raises(ValueError):
        nb_binormal_band(0.8, 0.05, 0.1, spread=0.03, side="upper")
