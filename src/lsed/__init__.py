"""lsed: prevalence-corrected evaluation of species-distribution predictions.

A Python mirror of the ``lsed`` R package: case weights that re-price an
evaluation sample to a target landscape prevalence, the prevalence-invariant
ROC-AUC, average-precision and net-benefit summaries (both the exact
point-level route and the equal-variance binormal route), and Cox calibration
of held-out predictions.
"""

from __future__ import annotations

from .average_precision import (
    ap_binormal,
    ap_pointwise,
    ap_scalar_to_prev,
    pr_curve_binormal,
    pr_curve_exact,
)
from .calibration import clamp, cox_calibration, recalibrate_to_prevalence
from .net_benefit import (
    nb_binormal,
    nb_binormal_band,
    nb_exact,
    nb_exact_ci,
    nb_treat_all,
)
from .reweight import reweight
from .roc import roc_auc

__version__ = "0.0.0.dev0"

__all__ = [
    "reweight",
    "roc_auc",
    "nb_exact",
    "nb_binormal",
    "nb_treat_all",
    "nb_exact_ci",
    "nb_binormal_band",
    "clamp",
    "cox_calibration",
    "recalibrate_to_prevalence",
    "ap_pointwise",
    "ap_binormal",
    "ap_scalar_to_prev",
    "pr_curve_exact",
    "pr_curve_binormal",
]
