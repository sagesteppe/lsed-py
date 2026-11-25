"""Average precision and precision-recall curves at a target prevalence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .reweight import reweight

__all__ = [
    "ap_pointwise",
    "pr_auc",
    "ap_binormal",
    "ap_scalar_to_prev",
    "pr_curve_exact",
    "pr_curve_binormal",
]


def _order_desc(prob):
    """Indices that order ``prob`` descending, ties keeping their input order."""
    return np.argsort(-prob, kind="stable")


def ap_pointwise(truth, prob, pi):
    """Exact per-point average precision (PR-AUC) at a target prevalence.

    Average precision computed directly from point-level predictions,
    case-weighted to a target landscape prevalence with :func:`reweight`.
    Recall is prevalence-invariant; precision is not, so re-pricing the weights
    re-prices the PR curve. This is the exact route, used whenever the full
    ``(truth, prob)`` pairs are available.

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 (presence) / 0 (absence).
    prob : array_like
        Predicted probabilities (or any score), same length as ``truth``.
    pi : float
        Target landscape prevalence in (0, 1).

    Returns
    -------
    float
        Average precision, or ``nan`` if a class is absent.

    See Also
    --------
    ap_binormal : the summary-metric counterpart.
    """
    truth = np.asarray(truth, dtype=float)
    prob = np.asarray(prob, dtype=float)
    n1 = np.sum(truth == 1)
    n0 = np.sum(truth == 0)
    if n1 == 0 or n0 == 0:
        return float("nan")
    w = reweight(truth, pi)
    o = _order_desc(prob)
    truth = truth[o]
    w = w[o]
    tpw = np.cumsum(w * (truth == 1))
    fpw = np.cumsum(w * (truth == 0))
    recall = tpw / pi                      # total presence weight = pi
    prec = tpw / (tpw + fpw)
    delta = np.concatenate(([recall[0]], np.diff(recall)))
    return float(np.sum(delta * prec))


def pr_auc(truth, prob):
    """Reported precision-recall AUC (average precision) at the sample prevalence.

    Average precision computed at the evaluation sample's own prevalence -- the
    ordinary (unweighted) PR-AUC a model reports from its test set, and the
    precision-side sibling of the prevalence-invariant :func:`lsed.roc_auc`. It is
    exactly :func:`ap_pointwise` evaluated at ``pi = mean(truth == 1)`` (where the
    prevalence weights become uniform); use :func:`ap_pointwise` directly to
    re-price average precision to a different target prevalence.

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 (presence) / 0 (absence).
    prob : array_like
        Predicted probabilities or scores, same length as ``truth``.

    Returns
    -------
    float
        Average precision at the sample prevalence, or ``nan`` if a class is
        absent.

    Notes
    -----
    Unlike :func:`lsed.roc_auc`, average precision is not inflated by easy
    absences, which is why it is the recommended anchor for curve reconstruction.
    This is still a per-sample *diagnostic*: to rebuild a prevalence-corrected
    decision curve, pass the reported value -- with the prevalence it was measured
    at -- to :func:`ap_scalar_to_prev` or :func:`lsed.nb_binormal_band`.

    See Also
    --------
    lsed.roc_auc, ap_pointwise, ap_scalar_to_prev, lsed.nb_binormal_band
    """
    truth = np.asarray(truth, dtype=float)
    n1 = np.sum(truth == 1)
    n0 = np.sum(truth == 0)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return ap_pointwise(truth, prob, pi=n1 / (n1 + n0))


def ap_binormal(mu, pi, grid=4000):
    """Average precision of an equal-variance binormal ROC at a prevalence.

    Average precision implied by an equal-variance binormal ROC with separation
    ``mu`` (so ``TPR = norm.cdf(norm.ppf(FPR) + mu)``), integrated at prevalence
    ``pi`` by the trapezoid rule. This is the summary-metric route: pair it with
    :func:`ap_scalar_to_prev` to turn a single reported PR-AUC into a full,
    prevalence-corrected PR curve shape.

    Parameters
    ----------
    mu : float
        Binormal separation (>= 0). Larger ``mu`` = better discrimination.
    pi : float
        Prevalence in (0, 1) at which to price precision.
    grid : int, default 4000
        Number of recall points for the trapezoid integral.

    Returns
    -------
    float
        Average precision.

    See Also
    --------
    ap_pointwise, ap_scalar_to_prev
    """
    f = np.linspace(1e-6, 1 - 1e-6, grid)   # FPR ascending -> recall ascending
    tpr = norm.cdf(norm.ppf(f) + mu)        # recall (invariant with prevalence)
    prec = (pi * tpr) / (pi * tpr + (1 - pi) * f)
    r = np.concatenate(([0.0], tpr))
    p = np.concatenate(([1.0], prec))
    return float(np.sum(np.diff(r) * (p[:-1] + p[1:]) / 2))   # trapezoid AP


def ap_scalar_to_prev(pr_auc, pi0, pi_target, grid=20000):
    """Correct a scalar PR-AUC from its design prevalence to a target prevalence.

    Given a reported PR-AUC measured at design prevalence ``pi0``, recover the
    equal-variance binormal ROC that reproduces it (:func:`ap_binormal`), then
    re-integrate precision at ``pi_target``. Use when only a summary PR-AUC and
    the prevalence it was measured at are available.

    Parameters
    ----------
    pr_auc : float
        Reported PR-AUC (average precision) at the design prevalence.
    pi0 : float
        Design/evaluation prevalence the ``pr_auc`` was measured at.
    pi_target : float
        Target landscape prevalence to re-price to.
    grid : int, default 20000
        Number of recall points passed to :func:`ap_binormal`.

    Returns
    -------
    dict
        ``{"ap_target": float, "mu": float, "implied_roc_auc": float}`` where
        ``implied_roc_auc = norm.cdf(mu / sqrt(2))``.
    """
    mu = brentq(lambda m: ap_binormal(m, pi0, grid=grid) - pr_auc, 1e-3, 8)
    return {
        "ap_target": ap_binormal(mu, pi_target, grid=grid),
        "mu": mu,
        "implied_roc_auc": float(norm.cdf(mu / np.sqrt(2))),
    }


def pr_curve_exact(truth, prob, pi):
    """Exact precision-recall curve at a target prevalence.

    Full PR curve from point-level predictions, case-weighted to prevalence
    ``pi`` with :func:`reweight`. The point-level analogue of
    :func:`ap_pointwise` (which integrates this curve).

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 / 0.
    prob : array_like
        Predicted probabilities / scores.
    pi : float
        Target prevalence in (0, 1).

    Returns
    -------
    pandas.DataFrame
        Columns ``recall`` and ``precision``, ordered by descending score.
    """
    truth = np.asarray(truth, dtype=float)
    prob = np.asarray(prob, dtype=float)
    w = reweight(truth, pi)
    o = _order_desc(prob)
    truth = truth[o]
    w = w[o]
    tpw = np.cumsum(w * (truth == 1))
    fpw = np.cumsum(w * (truth == 0))
    return pd.DataFrame({"recall": tpw / pi, "precision": tpw / (tpw + fpw)})


def pr_curve_binormal(mu, pi, grid=400):
    """Reconstructed precision-recall curve from a binormal ROC at a prevalence.

    PR curve implied by an equal-variance binormal ROC (separation ``mu``) at
    prevalence ``pi``. The point-level analogue of :func:`ap_binormal` and the
    summary-metric counterpart of :func:`pr_curve_exact`.

    Parameters
    ----------
    mu : float
        Binormal separation (>= 0).
    pi : float
        Prevalence in (0, 1).
    grid : int, default 400
        Number of recall points on the curve.

    Returns
    -------
    pandas.DataFrame
        Columns ``recall`` and ``precision``.
    """
    r = np.linspace(1e-3, 1 - 1e-3, grid)
    fpr = norm.cdf(norm.ppf(r) - mu)        # invert TPR = norm.cdf(norm.ppf(FPR) + mu)
    o = pi / (1 - pi)
    return pd.DataFrame({"recall": r, "precision": (o * r) / (o * r + fpr)})
