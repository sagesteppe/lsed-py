"""ROC-AUC via the rank (Mann-Whitney) identity."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

__all__ = ["roc_auc"]


def roc_auc(truth, prob):
    """Area under the ROC curve from the rank-sum of the positive scores.

    Equivalent to the probability that a random presence outscores a random
    absence (ties counting as one half). Prevalence-invariant, so no case
    weighting is needed.

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 (presence) / 0 (absence).
    prob : array_like
        Predicted probabilities or scores, same length as ``truth``.

    Returns
    -------
    float
        ROC-AUC, or ``nan`` if a class is absent.
    """
    truth = np.asarray(truth, dtype=float)
    prob = np.asarray(prob, dtype=float)
    if np.unique(truth).size < 2:
        return float("nan")
    r = rankdata(prob)  # average ranks for ties, matching R's rank()
    n1 = np.sum(truth == 1)
    n0 = np.sum(truth == 0)
    return (np.sum(r[truth == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
