"""Decision-curve net benefit at a target prevalence."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .reweight import reweight

__all__ = ["nb_exact", "nb_binormal", "nb_treat_all"]


def nb_exact(truth, prob, pi, pt):
    r"""Net benefit of thresholded predictions at a target prevalence.

    Decision-curve net benefit of treating (surveying) every unit whose
    predicted probability is at least the threshold ``pt``, case-weighted to a
    target landscape prevalence with :func:`reweight`:

    .. math:: NB(p_t) = TP_w - FP_w \cdot \frac{p_t}{1 - p_t},

    where :math:`TP_w` and :math:`FP_w` are the prevalence-weighted true- and
    false-positive fractions. This is the calibration-*dependent* (per-point)
    route: it thresholds the probabilities as given.

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 (presence) / 0 (absence).
    prob : array_like
        Predicted probabilities, same length as ``truth``.
    pi : float
        Target landscape prevalence in (0, 1).
    pt : float or array_like
        Threshold probability or probabilities.

    Returns
    -------
    numpy.ndarray
        Net benefit, one value per element of ``pt``.

    See Also
    --------
    nb_binormal : best-achievable net benefit from an ROC.
    nb_treat_all : the treat-all reference.
    """
    truth = np.asarray(truth, dtype=float)
    prob = np.asarray(prob, dtype=float)
    pt = np.atleast_1d(np.asarray(pt, dtype=float))
    w = reweight(truth, pi)
    n = np.sum(w)

    def _nb(t):
        pp = prob >= t
        return (np.sum(w[pp & (truth == 1)]) / n
                - np.sum(w[pp & (truth == 0)]) / n * (t / (1 - t)))

    return np.array([_nb(t) for t in pt])


def nb_binormal(mu, pi, pt):
    """Best-achievable net benefit from an equal-variance binormal ROC.

    Net benefit read off an equal-variance binormal ROC (separation ``mu``) at
    the operating point whose slope equals the cost ratio
    ``((1 - pi) / pi) * (pt / (1 - pt))``. Because it always uses the optimal
    cutoff, this is the calibration-*independent*, best-achievable net benefit
    for that ROC -- the summary-metric counterpart of :func:`nb_exact`.

    Parameters
    ----------
    mu : float
        Binormal separation (>= 0).
    pi : float
        Target landscape prevalence in (0, 1).
    pt : float or array_like
        Threshold probability or probabilities.

    Returns
    -------
    numpy.ndarray
        Net benefit, one value per element of ``pt``.
    """
    pt = np.atleast_1d(np.asarray(pt, dtype=float))
    m = ((1 - pi) / pi) * (pt / (1 - pt))   # target ROC slope for each threshold
    cc = (np.log(m) + mu ** 2 / 2) / mu     # score cutoff giving that slope
    return pi * norm.cdf(mu - cc) - (1 - pi) * norm.cdf(-cc) * (pt / (1 - pt))


def nb_treat_all(pi, pt):
    """Net benefit of the treat-all (survey-everything) reference strategy.

    The decision-curve reference that acts on every unit:
    ``pi - (1 - pi) * (pt / (1 - pt))``. Treat-none is identically 0.

    Parameters
    ----------
    pi : float
        Prevalence in (0, 1).
    pt : float or array_like
        Threshold probability or probabilities.

    Returns
    -------
    numpy.ndarray
        Net benefit, one value per element of ``pt``.
    """
    pt = np.atleast_1d(np.asarray(pt, dtype=float))
    return pi - (1 - pi) * (pt / (1 - pt))
