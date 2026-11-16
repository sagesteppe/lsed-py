"""Prevalence case weights."""

from __future__ import annotations

import numpy as np

__all__ = ["reweight"]


def reweight(truth, pi):
    """Case weights that re-price an evaluation sample to a target prevalence.

    Every presence receives weight ``pi / n1`` and every absence
    ``(1 - pi) / n0``, so the total presence weight is ``pi``, the total absence
    weight is ``1 - pi``, and the weights sum to 1. Thresholded summaries
    (recall, precision, net benefit) computed with these weights behave as if
    the sample had been drawn at prevalence ``pi``.

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 (presence) / 0 (absence).
    pi : float
        Target landscape prevalence in (0, 1).

    Returns
    -------
    numpy.ndarray
        Weights, the same length as ``truth``.

    Examples
    --------
    >>> w = reweight([1, 0, 0, 1], pi=0.05)
    >>> float(w.sum())
    1.0
    """
    truth = np.asarray(truth, dtype=float)
    n1 = np.sum(truth == 1)
    n0 = np.sum(truth == 0)
    return np.where(truth == 1, pi / n1, (1 - pi) / n0)
