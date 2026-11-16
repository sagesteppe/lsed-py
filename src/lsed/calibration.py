"""Calibration helpers for held-out predictions."""

from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit

__all__ = ["clamp", "cox_calibration", "recalibrate_to_prevalence"]


def clamp(p, eps=1e-6):
    """Bound probabilities into ``[eps, 1 - eps]`` so logit transforms stay finite.

    Used by the calibration helpers before :func:`scipy.special.logit`.

    Parameters
    ----------
    p : array_like
        Probabilities.
    eps : float, default 1e-6
        Small margin; values are clamped to ``[eps, 1 - eps]``.

    Returns
    -------
    numpy.ndarray
        Clamped probabilities, the same length as ``p``.
    """
    p = np.asarray(p, dtype=float)
    return np.minimum(np.maximum(p, eps), 1 - eps)


def _glm_binomial(endog, exog, offset=None):
    """Fit a binomial GLM, mirroring R's ``glm(..., family = binomial)``.

    Returns ``(params, converged)`` or ``(None, False)`` if the fit errors out
    (e.g. perfect separation).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.GLM(endog, exog, family=sm.families.Binomial(),
                         offset=offset).fit()
        except Exception:
            return None, False
    return np.asarray(fit.params, dtype=float), bool(fit.converged)


def cox_calibration(truth, prob):
    """Cox (logistic) calibration intercept and slope on held-out predictions.

    The slope comes from ``glm(truth ~ logit(prob))``; the intercept
    ("calibration-in-the-large") from ``glm(truth ~ 1, offset = logit(prob))``.
    Separated or non-converging fits (``|coef| >= 15``) are dropped to ``nan``.

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 / 0.
    prob : array_like
        Predicted probabilities.

    Returns
    -------
    dict
        ``{"cox_intercept": float, "cox_slope": float}``; either entry may be
        ``nan``.
    """
    truth = np.asarray(truth, dtype=float)
    lp = logit(clamp(prob))

    # slope: truth ~ 1 + lp
    b_slope, conv_slope = _glm_binomial(truth, sm.add_constant(lp, has_constant="add"))
    # intercept: truth ~ 1, with lp as an offset
    b_intc, conv_intc = _glm_binomial(truth, np.ones((truth.size, 1)), offset=lp)

    def _ok(b, conv):
        return b is not None and conv and np.all(np.abs(b) < 15)

    return {
        "cox_intercept": float(b_intc[0]) if _ok(b_intc, conv_intc) else float("nan"),
        "cox_slope": float(b_slope[1]) if _ok(b_slope, conv_slope) else float("nan"),
    }


def recalibrate_to_prevalence(prob, p_obs, pi):
    """Recalibrate predicted probabilities to a target prevalence (logit shift).

    Shift predictions on the logit scale from their observed prevalence
    ``p_obs`` to a target prevalence ``pi``, leaving relative ordering intact:
    ``expit(logit(prob) - logit(p_obs) + logit(pi))``. Inputs are clamped with
    :func:`clamp` to keep the logits finite.

    Parameters
    ----------
    prob : array_like
        Predicted probabilities.
    p_obs : float
        Prevalence the predictions were produced at.
    pi : float
        Target landscape prevalence in (0, 1).

    Returns
    -------
    numpy.ndarray
        Recalibrated probabilities.
    """
    return expit(logit(clamp(prob)) - logit(clamp(p_obs)) + logit(pi))
