"""Survey-intensity field for the S term.

A Python mirror of ``fn_survey.R``: a gridless kernel model of how thoroughly
ground has been botanised. Records are weighted (plots vs opportunistic, with
inverse-frequency species weights and a basis multiplier), then smeared by a
truncated Gaussian so :func:`eval_survey` can report survey intensity at any
point -- the patch centroids, for the ranking's S term.

R's ``eval_survey`` hand-rolls a ``trunc*sigma`` bucket grid for locality; here
a :class:`scipy.spatial.cKDTree` ball query does the same neighbour restriction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

__all__ = [
    "survey_weights",
    "survey_field",
    "eval_survey",
    "survey_term",
    "survey_raster",
    "guess_record_type",
    "SurveyField",
]

#: Default regex patterns marking a vegetation-plot data source (case-insensitive).
PLOT_PATTERNS = (
    "vegbank",
    r"\bFIA\b",
    r"\bAIM\b",
    r"\bNEON\b",
    r"\bCVS\b",
    "carolina veg",
    r"\bNPS\b",
    "rel[eé]v[eé]",
    "quadrat",
    r"\bplot\b",
    "inventory",
)


@dataclass
class SurveyField:
    """Bundled records, per-record weights and bandwidth (see :func:`survey_field`)."""

    coords: np.ndarray  # (n, 2)
    species: np.ndarray  # (n,), plot points labelled "__plot__"
    w: np.ndarray  # (n,) per-record weight
    sigma: float
    trunc: float
    plot_weight: float
    weights: "pd.DataFrame | None" = _field(default=None)


def _union(domain):
    """Coerce a domain (shapely geom, GeoSeries, GeoDataFrame, iterable) to one geom."""
    from shapely.ops import unary_union

    geoms = getattr(domain, "geometry", domain)
    if hasattr(geoms, "geom_type"):  # already a single shapely geometry
        return geoms
    return unary_union(list(geoms))


def survey_weights(coords, species, domain=None, min_records=5, p=0.95, damp=1.0):
    """Inverse-frequency per-species weights, range-corrected by domain overlap.

    A widespread species contributes many records but little local survey
    evidence; the weight ``(phi / n) ** damp`` down-weights common taxa, where
    ``phi`` is the share of the species' central ``p`` bounding box that falls
    inside ``domain`` (1 when no domain, or for a degenerate box).

    Parameters
    ----------
    coords : array_like, shape (n, 2)
        Record coordinates (metres).
    species : array_like
        Species label per record.
    domain : optional
        Study-area polygon(s) (shapely / GeoSeries / GeoDataFrame).
    min_records : int, optional
        Species with fewer records are dropped (default 5).
    p : float, optional
        Central fraction for the bounding box (default 0.95).
    damp : float, optional
        Power on the raw weight (default 1; 0 makes every species equal).

    Returns
    -------
    pandas.DataFrame
        One row per qualifying species (``species, n, phi, freq, w``), sorted by
        species name.
    """
    from shapely.geometry import box as _box

    coords = np.asarray(coords, dtype=float)[:, :2]
    species = np.asarray(species).astype(str)
    if len(species) != len(coords):
        raise ValueError("species must have one label per record")
    lo = (1 - p) / 2
    qs = [lo, 1 - lo]
    dom = _union(domain) if domain is not None else None

    uniq = sorted(set(species))
    rows = []
    for sp in uniq:
        idx = np.where(species == sp)[0]
        n = len(idx)
        if n < min_records:
            continue
        xy = coords[idx]
        bx = np.quantile(xy[:, 0], qs)  # type-7 linear == numpy default
        by = np.quantile(xy[:, 1], qs)
        area_box = (bx[1] - bx[0]) * (by[1] - by[0])
        phi = 1.0
        if dom is not None and area_box > 0:
            inter = _box(bx[0], by[0], bx[1], by[1]).intersection(dom).area
            phi = min(max(inter / area_box, 0.0), 1.0)
        freq = n / max(phi, np.finfo(float).eps)
        rows.append(dict(species=sp, n=n, phi=phi, freq=freq, w=(phi / n) ** damp))
    if not rows:
        raise ValueError(f"no species clears min_records = {min_records}")
    return pd.DataFrame(rows)


def survey_field(
    coords,
    species,
    sigma=100.0,
    record_type=None,
    plot_types=("Plot",),
    plot_weight=1.0,
    basis=None,
    basis_weights=None,
    weights=None,
    domain=None,
    min_records=5,
    p=0.95,
    damp=1.0,
    trunc=4.0,
):
    """Construct the gridless survey-intensity field.

    Splits the two detection processes: ``record_type`` labels in ``plot_types``
    are vegetation plots (deduplicated by coordinate, held out of the frequency
    counts, each carrying ``plot_weight``); the rest are opportunistic records
    weighted by :func:`survey_weights` times an optional ``basis`` multiplier.

    Parameters
    ----------
    coords : array_like, shape (n, 2)
        Record coordinates (metres).
    species : array_like
        Species label per record.
    sigma : float, optional
        Kernel bandwidth in metres (default 100).
    record_type : array_like, optional
        Per-record label; those in ``plot_types`` are plots. None = all opportunistic.
    plot_types : iterable of str, optional
        Labels marking a plot (case-insensitive; default ``("Plot",)``).
    plot_weight : float, optional
        Weight per deduplicated plot point (default 1).
    basis : array_like, optional
        Per-record basis-of-record labels, to up-weight deliberate collections.
    basis_weights : dict, optional
        Per-basis multipliers (default ``{"PRESERVED_SPECIMEN": 1.5}``).
    weights : pandas.DataFrame or dict, optional
        Per-species weights (as from :func:`survey_weights`); None computes them.
    domain, min_records, p, damp : optional
        Passed to :func:`survey_weights` when ``weights`` is None.
    trunc : float, optional
        Kernel support in units of sigma (default 4).

    Returns
    -------
    SurveyField
    """
    basis_weights = {"PRESERVED_SPECIMEN": 1.5} if basis_weights is None else basis_weights
    coords = np.asarray(coords, dtype=float)[:, :2]
    species = np.asarray(species).astype(str)
    if sigma <= 0 or trunc <= 0:
        raise ValueError("sigma and trunc must be positive")
    if basis is not None:
        basis = np.asarray(basis).astype(object)

    is_plot = np.zeros(len(coords), dtype=bool)
    if record_type is not None:
        rt = np.asarray(record_type).astype(str)
        pt = {s.lower() for s in plot_types}
        is_plot = np.array([s.lower() in pt for s in rt])

    # plot stream: one survey event per location -> dedup on coords
    pc = np.empty((0, 2))
    pw = np.empty(0)
    psp = np.empty(0, dtype=object)
    if is_plot.any():
        cp = np.unique(coords[is_plot], axis=0)
        pc = cp
        pw = np.full(len(cp), float(plot_weight))
        psp = np.array(["__plot__"] * len(cp), dtype=object)

    # opportunistic stream: inverse-frequency weights (plots excluded from counts)
    oc = coords[~is_plot]
    osp = species[~is_plot]
    ow = np.empty(0)
    if len(oc):
        if weights is None:
            weights = survey_weights(
                oc, osp, domain=domain, min_records=min_records, p=p, damp=damp
            )
        if isinstance(weights, pd.DataFrame):
            wmap = dict(zip(weights["species"], weights["w"]))
        else:
            wmap = dict(weights)
        keep = np.array([s in wmap for s in osp])
        oc = oc[keep]
        osp = osp[keep]
        ow = np.array([wmap[s] for s in osp], dtype=float)
        if basis is not None:
            bo = basis[~is_plot][keep]
            mult = np.array([basis_weights.get(b, 1.0) for b in bo], dtype=float)
            ow = ow * mult

    out_coords = np.vstack([oc, pc]) if len(oc) or len(pc) else np.empty((0, 2))
    out_species = np.concatenate([osp.astype(object), psp])
    w = np.concatenate([ow, pw])
    if not len(w):
        raise ValueError("no records remain after record types / the floor")

    return SurveyField(
        coords=out_coords,
        species=out_species,
        w=w,
        sigma=float(sigma),
        trunc=float(trunc),
        plot_weight=float(plot_weight),
        weights=weights if isinstance(weights, pd.DataFrame) else None,
    )


def eval_survey(field, xy, species=None):
    """Evaluate the survey-intensity field ``S(x)`` at query points.

    ``S(x) = sum_j w_j * exp(-||x - c_j||^2 / (2 sigma^2))`` over records within
    ``trunc * sigma`` of ``x``. Gridless, so it reports intensity at arbitrary
    points (the patch centroids) without rendering a raster.

    Parameters
    ----------
    field : SurveyField
    xy : array_like, shape (m, 2)
        Query coordinates (same CRS as the records).
    species : iterable, optional
        Restrict the sum to these species labels.

    Returns
    -------
    numpy.ndarray
        ``S`` at each query point, length ``m``.
    """
    xy = np.asarray(xy, dtype=float)[:, :2]
    cd = field.coords
    wd = field.w
    if species is not None:
        keep = np.isin(field.species, list(species))
        cd = cd[keep]
        wd = wd[keep]

    out = np.zeros(len(xy))
    if not len(wd) or not len(xy):
        return out

    bw = field.trunc * field.sigma
    denom = 2 * field.sigma**2
    tree = cKDTree(cd)
    neighbours = tree.query_ball_point(xy, r=bw)  # local support, like R's buckets
    for qi, cand in enumerate(neighbours):
        if not cand:
            continue
        cand = np.asarray(cand)
        d2 = np.sum((xy[qi] - cd[cand]) ** 2, axis=1)
        out[qi] = np.sum(np.exp(-d2 / denom) * wd[cand])
    return out


def survey_term(patches, field, species=None):
    """The S ranking term at patch centroids: ``-eval_survey`` (well-surveyed = low).

    Returns a vector aligned to ``patches`` order, ready to drop into
    :func:`lsed.rank.rank_calculate` as ``extra_terms={"S": ...}`` -- negated so
    the rescale makes well-surveyed ground low priority and unsurveyed high.
    """
    cen = patches.geometry.centroid
    xy = np.column_stack([cen.x.to_numpy(), cen.y.to_numpy()])
    return -eval_survey(field, xy, species=species)


def guess_record_type(origin, plot_label="Plot", other_label="Opportunistic",
                      plot_patterns=PLOT_PATTERNS):
    """Classify records as plot vs opportunistic from a dataset/origin string.

    Case-insensitive regex match against known vegetation-plot sources
    (FIA, AIM, NEON, VegBank, ...); a hit is a plot. Mirrors
    ``guess_record_type`` in ``fn_survey.R``. ``None`` origins stay ``None``.

    Parameters
    ----------
    origin : iterable of str
        Per-record dataset / origin labels.
    plot_label, other_label : str, optional
        Labels returned for a match / non-match.
    plot_patterns : iterable of str, optional
        Regex patterns marking a plot source (default :data:`PLOT_PATTERNS`).

    Returns
    -------
    list
        One label per record (``None`` where the origin was ``None``).
    """
    rx = re.compile("|".join(plot_patterns), re.IGNORECASE)
    out = []
    for o in origin:
        if o is None or (isinstance(o, float) and np.isnan(o)):
            out.append(None)
        else:
            out.append(plot_label if rx.search(str(o)) else other_label)
    return out


def survey_raster(field, transform, shape, rescale="quantile", q=0.95, invert=False,
                  species=None):
    """Render the survey field to a raster array (R's ``survey_raster``).

    Evaluates the gridless field at every cell centre and optionally rescales:
    ``"max"`` divides by the max, ``"quantile"`` by the ``q`` quantile then
    clamps to ``[0, 1]``, ``"none"`` leaves raw intensity. The ranking uses the
    gridless :func:`eval_survey` directly; this is for mapping / inspection.

    Parameters
    ----------
    field : SurveyField
    transform : affine.Affine
        Target grid georeferencing.
    shape : tuple of int
        ``(rows, cols)``.
    rescale : {"quantile", "max", "none"}, optional
        Normalisation (default "quantile").
    q : float, optional
        Quantile for ``rescale="quantile"`` (default 0.95).
    invert : bool, optional
        Return ``1 - S`` (needs a bounded rescale).
    species : iterable, optional
        Restrict to these species.

    Returns
    -------
    numpy.ndarray
        The ``S`` surface of shape ``shape``.
    """
    if invert and rescale == "none":
        raise ValueError("invert=True needs rescale='max' or 'quantile'")
    nr, nc = shape
    cols, rows = np.meshgrid(np.arange(nc), np.arange(nr))
    c = cols.ravel() + 0.5
    r = rows.ravel() + 0.5
    a, b, xoff, d, e, yoff = transform[:6]
    xs = a * c + b * r + xoff  # cell-centre map coords
    ys = d * c + e * r + yoff
    s = eval_survey(field, np.column_stack([xs, ys]), species=species).reshape(nr, nc)

    if rescale != "none":
        denom = np.nanmax(s) if rescale == "max" else np.nanquantile(s, q)
        if np.isfinite(denom) and denom > 0:
            s = np.clip(s / denom, 0, 1)
            if invert:
                s = 1 - s
    return s
