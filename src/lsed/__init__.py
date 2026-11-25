"""lsed: prevalence-corrected evaluation of species-distribution predictions.

A Python mirror of the ``lsed`` R package. Two halves: prevalence-corrected
*evaluation* of species-distribution predictions (case weights, prevalence-
invariant ROC-AUC, average-precision and net-benefit summaries, Cox
calibration), and the spatial *prioritisation* engine (patch/subpatch
delineation helpers and a road/trail travel-time network answering the
accessibility and complementarity terms).
"""

from __future__ import annotations

from .average_precision import (
    ap_binormal,
    ap_pointwise,
    ap_scalar_to_prev,
    pr_auc,
    pr_curve_binormal,
    pr_curve_exact,
)
from .calibration import clamp, cox_calibration, recalibrate_to_prevalence
from .contract import smooth_degree2
from .net_benefit import (
    nb_binormal,
    nb_binormal_band,
    nb_exact,
    nb_exact_ci,
    nb_treat_all,
)
from .network import (
    accessibility,
    build_network,
    complementarity,
    hop_speed,
    osm_speeds,
    rescale01,
)
from .patches import comp_labels, kcap, rook_edges, snap_pieces
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
    "pr_auc",
    "ap_binormal",
    "ap_scalar_to_prev",
    "pr_curve_exact",
    "pr_curve_binormal",
    # spatial prioritisation: patch helpers
    "rook_edges",
    "comp_labels",
    "snap_pieces",
    "kcap",
    "smooth_degree2",
    # spatial prioritisation: travel-time network
    "osm_speeds",
    "rescale01",
    "hop_speed",
    "build_network",
    "accessibility",
    "complementarity",
]
