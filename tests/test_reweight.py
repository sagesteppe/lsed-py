import numpy as np

from lsed import reweight


def test_weights_sum_to_one():
    truth = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    assert reweight(truth, pi=0.05).sum() == 1
    assert reweight(truth, pi=0.5).sum() == 1


def test_class_totals_equal_the_prevalence_split():
    truth = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    pi = 0.05
    w = reweight(truth, pi)
    assert np.isclose(w[truth == 1].sum(), pi)
    assert np.isclose(w[truth == 0].sum(), 1 - pi)


def test_every_case_in_a_class_gets_the_same_weight():
    truth = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    pi = 0.2
    n1 = np.sum(truth == 1)
    n0 = np.sum(truth == 0)
    w = reweight(truth, pi)
    assert np.all(w[truth == 1] == pi / n1)
    assert np.all(w[truth == 0] == (1 - pi) / n0)


def test_output_aligned_to_input_length_and_order():
    truth = np.array([1, 0, 1])
    w = reweight(truth, 0.3)
    assert len(w) == len(truth)
    o = [2, 0, 1]
    assert np.allclose(reweight(truth[o], 0.3), w[o])
