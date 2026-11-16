import numpy as np

from lsed import nb_binormal, nb_exact, nb_treat_all


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
