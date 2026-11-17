"""Decision-curve net benefit at a target prevalence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .average_precision import ap_binormal
from .reweight import reweight

__all__ = [
    "nb_exact",
    "nb_binormal",
    "nb_treat_all",
    "nb_exact_ci",
    "nb_binormal_band",
]


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


def nb_exact_ci(truth, prob, pi, pt, B=999, level=0.8, side="lower",
                stratified=True, random_state=None):
    """Bootstrap confidence band for the exact net-benefit curve.

    Percentile bootstrap confidence limits for :func:`nb_exact`, following the
    extensions of Vickers et al. (2008). Resampling is *stratified by outcome* by
    default: presences and absences are resampled with replacement separately so
    :math:`n_1` and :math:`n_0` -- and therefore the prevalence case weights
    (:func:`reweight`) -- stay fixed. This keeps the interval consistent with the
    prevalence-invariant estimand, propagating only the sampling noise in the
    within-class positive rates rather than in the sample's own prevalence.

    Only a *lower* band is returned by default. In decision-curve analysis the
    operative question is whether the model is at least as good as the best
    default strategy (treat-all or treat-none), so the decision-relevant
    uncertainty is the downside; an upper limit changes no decision. Pass
    ``side="both"`` for a two-sided interval.

    The bootstrap is vectorised over all ``B`` replicates with a single
    count-matrix multiply (the R mirror parallelises replicates instead); pass
    ``random_state`` for reproducibility.

    Parameters
    ----------
    truth : array_like
        Outcomes coded 1 (presence) / 0 (absence).
    prob : array_like
        Predicted probabilities, same length as ``truth``. Assumed honest
        (held-out or cross-validated); the band quantifies precision, not
        optimism, and applies no overfit correction.
    pi : float
        Target landscape prevalence in (0, 1).
    pt : float or array_like
        Threshold probability or probabilities.
    B : int, default 999
        Number of bootstrap replicates.
    level : float, default 0.8
        Confidence level.
    side : {"lower", "both"}, default "lower"
        One-sided lower band, or a two-sided interval.
    stratified : bool, default True
        Resample within outcome classes (True) or over all observations (False).
    random_state : int, numpy.random.Generator, or None
        Seed or generator for reproducibility.

    Returns
    -------
    pandas.DataFrame
        One row per ``pt``: ``pt``, the point estimate ``nb``, and ``lower``
        (plus ``upper`` when ``side="both"``).

    See Also
    --------
    nb_exact, nb_binormal_band

    References
    ----------
    Vickers AJ, Cronin AM, Elkin EB, Gonen M (2008). Extensions to decision
    curve analysis. *BMC Med Inform Decis Mak* 8:53.
    """
    if side not in ("lower", "both"):
        raise ValueError("side must be 'lower' or 'both'")
    truth = np.asarray(truth, dtype=float)
    prob = np.asarray(prob, dtype=float)
    pt = np.atleast_1d(np.asarray(pt, dtype=float))
    rng = np.random.default_rng(random_state)

    est = nb_exact(truth, prob, pi, pt)
    cost = pt / (1 - pt)                              # cost ratio per threshold
    ind = (prob[:, None] >= pt[None, :]).astype(float)   # (n, K) exceedances
    pos = (truth == 1).astype(float)
    neg = (truth == 0).astype(float)

    if stratified:
        ip = ind[truth == 1]                         # (n1, K)
        in_ = ind[truth == 0]                        # (n0, K)
        n1, n0 = ip.shape[0], in_.shape[0]
        c_pos = rng.multinomial(n1, np.full(n1, 1 / n1), size=B)   # (B, n1)
        c_neg = rng.multinomial(n0, np.full(n0, 1 / n0), size=B)   # (B, n0)
        tpr = (c_pos @ ip) / n1                       # (B, K)
        fpr = (c_neg @ in_) / n0
        boot = pi * tpr - (1 - pi) * fpr * cost[None, :]
    else:
        n = truth.size
        c = rng.multinomial(n, np.full(n, 1 / n), size=B)          # (B, n)
        n1_b = (c @ pos)[:, None]
        n0_b = (c @ neg)[:, None]
        tp = c @ (ind * pos[:, None])                 # (B, K)
        fp = c @ (ind * neg[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = pi * tp / n1_b - (1 - pi) * fp / n0_b * cost[None, :]

    a = 1 - level
    if side == "lower":
        lower = np.nanquantile(boot, a, axis=0)
        return pd.DataFrame({"pt": pt, "nb": est, "lower": lower})
    lower = np.nanquantile(boot, a / 2, axis=0)
    upper = np.nanquantile(boot, 1 - a / 2, axis=0)
    return pd.DataFrame({"pt": pt, "nb": est, "lower": lower, "upper": upper})


def nb_binormal_band(pr_auc, pi, pt, spread, side="lower", pi0=None, grid=3000):
    """Plausibility band for a binormal net-benefit curve from PR-AUC uncertainty.

    The summary route (:func:`nb_binormal`) anchors an entire decision curve to a
    single reported PR-AUC, almost always a point estimate with no stated
    uncertainty. This propagates a plausible *spread* in that PR-AUC into a band
    on the net-benefit curve, extending :func:`ap_scalar_to_prev` from a point
    reconstruction to an interval one.

    The perturbation is anchored on the PR-AUC scale, not on ``mu`` directly: a
    fixed shift in ``mu`` has no stable meaning across taxa or prevalences,
    whereas a spread in PR-AUC is what replicate train/test partitions actually
    show. The reported PR-AUC and its ``spread``-perturbed endpoints are each
    converted to a binormal separation by inverting :func:`ap_binormal` at the
    evaluation prevalence ``pi0``, then read off :func:`nb_binormal` at the target
    prevalence. Because ``nb_binormal`` is monotone increasing in ``mu``, the
    PR-AUC endpoints map straight to the band edges.

    This is a *plausibility* band (a sensitivity analysis over an assumed PR-AUC
    spread), not a confidence interval: with only a reported scalar there is no
    sampling distribution. Calibrate ``spread`` from replicate models when you
    have them; see :func:`nb_exact_ci` for the model-free sampling band when raw
    predictions are available.

    Only a *lower* band is returned by default, matching :func:`nb_exact_ci`: the
    decision-relevant uncertainty is the downside. Pass ``side="both"`` for upper
    and lower edges.

    Parameters
    ----------
    pr_auc : float
        Reported PR-AUC (average precision) at the evaluation prevalence.
    pi : float
        Target landscape prevalence in (0, 1).
    pt : float or array_like
        Threshold probability or probabilities.
    spread : float
        Half-width of the plausible PR-AUC range, in PR-AUC units (e.g. ``0.04``
        for +/- 0.04). Calibrate from replicate-model spread.
    side : {"lower", "both"}, default "lower"
        One-sided lower band, or lower and upper edges.
    pi0 : float, optional
        Evaluation prevalence the PR-AUC was measured at. Defaults to ``pi``.
    grid : int, default 3000
        Integration grid passed to :func:`ap_binormal` when inverting the PR-AUC.

    Returns
    -------
    pandas.DataFrame
        One row per ``pt``: ``pt``, the central curve ``nb``, and ``lower`` (plus
        ``upper`` when ``side="both"``). The fitted separations are stored in
        ``df.attrs["mu"]`` (keys ``central``, ``lower``, and ``upper`` when
        two-sided).

    See Also
    --------
    nb_binormal, nb_exact_ci, ap_scalar_to_prev
    """
    if side not in ("lower", "both"):
        raise ValueError("side must be 'lower' or 'both'")
    if pi0 is None:
        pi0 = pi
    pt = np.atleast_1d(np.asarray(pt, dtype=float))

    def to_mu(val):
        val = min(max(val, 1e-4), 1 - 1e-4)
        return brentq(lambda m: ap_binormal(m, pi0, grid=grid) - val, 1e-3, 8)

    mu = {"central": to_mu(pr_auc), "lower": to_mu(pr_auc - spread)}
    data = {
        "pt": pt,
        "nb": nb_binormal(mu["central"], pi, pt),
        "lower": nb_binormal(mu["lower"], pi, pt),
    }
    if side == "both":
        mu["upper"] = to_mu(pr_auc + spread)
        data["upper"] = nb_binormal(mu["upper"], pi, pt)
    out = pd.DataFrame(data)
    out.attrs["mu"] = mu
    return out
