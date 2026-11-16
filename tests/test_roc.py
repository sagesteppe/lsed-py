import numpy as np

from lsed import roc_auc


def test_perfect_reversed_and_uninformative_rankings():
    truth = np.array([0, 0, 1, 1])
    assert roc_auc(truth, [1, 2, 3, 4]) == 1      # presences outscore
    assert roc_auc(truth, [4, 3, 2, 1]) == 0      # absences outscore
    assert roc_auc(truth, [1, 1, 1, 1]) == 0.5    # all tied


def test_missing_class_returns_nan():
    assert np.isnan(roc_auc(np.ones(5), np.random.rand(5)))
    assert np.isnan(roc_auc(np.zeros(5), np.random.rand(5)))


def test_rank_identity_equals_bruteforce_concordance():
    # AUC = P(presence scores above absence), ties counting as 1/2.
    rng = np.random.default_rng(1)
    truth = rng.binomial(1, 0.4, size=60)
    prob = rng.random(60)
    pos = prob[truth == 1]
    neg = prob[truth == 0]
    conc = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    assert np.isclose(roc_auc(truth, prob), conc.mean())


def test_invariant_to_monotonic_transform():
    rng = np.random.default_rng(2)
    truth = rng.binomial(1, 0.5, size=50)
    prob = rng.random(50)
    logit = np.log(prob / (1 - prob))
    assert np.isclose(roc_auc(truth, prob), roc_auc(truth, logit))
