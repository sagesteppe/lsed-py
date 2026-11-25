"""Combine the RankIndex terms into a patch priority score.

A Python mirror of ``fn_rank.R`` (:func:`rank_index`) and the orchestration
half of ``fn_calculate.R`` (:func:`rank_calculate`).

:func:`rank_index` is the pure combiner. :func:`rank_calculate` assembles the
operational terms end-to-end: it computes accessibility (A) and complementarity
(C) from a road/trail network (via :mod:`lsed.network`), reads the patch
suitability term (P) from a column, and accepts the raster-derived terms
(Dg, Dr, S) as precomputed per-patch vectors -- their raster computation
(R's ``terra``/``whitebox`` surfaces) needs ``rasterio`` and is not ported here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .network import accessibility, build_network, complementarity, osm_speeds
from .patches import cluster_patches

__all__ = ["rank_index", "rank_calculate"]


def _ac_block(net_lines, patches_sub, origin_xy, areas_sub, speeds,
              offtrail_speed, offtrail_penalty, target_area, cap_q, invert):
    """Accessibility + complementarity for one patch subset on one sub-network."""
    net = build_network(net_lines, speeds=speeds)
    ab = accessibility(
        net, patches_sub, origins_xy=origin_xy,
        offtrail_speed=offtrail_speed, offtrail_penalty=offtrail_penalty, invert=invert,
    )
    comp = complementarity(
        ab["net"], areas=areas_sub, target_area=target_area, cap_q=cap_q, invert=invert
    )
    return ab["access_hr"], comp["assemble_hr"]


def _rescale01(x):
    """Min-max to [0, 1] over finite values; constant -> 0, all-NaN -> NaN."""
    x = np.asarray(x, dtype=float)
    fin = x[np.isfinite(x)]
    if fin.size == 0:
        return x * 0.0  # all-NaN stays NaN
    lo, hi = fin.min(), fin.max()
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


def rank_index(terms, weights=None, rescale=True):
    """Combine named per-patch term vectors into a priority index and rank.

    Assembles ``Rank(x) = Dg + Dr + S + A + C`` from whichever terms are given.
    Each term is min-max rescaled to ``[0, 1]`` (unless ``rescale=False``), then
    combined as a per-patch weighted mean over the terms present -- so the index
    stays in ``[0, 1]`` however many terms are supplied, and a patch missing a
    term is scored on its remaining terms (a patch missing every term is NaN).

    Parameters
    ----------
    terms : dict of str -> array_like
        Named per-patch term vectors (e.g. ``{"Dg": ..., "A": ...}``), all equal
        length, patches in the same order.
    weights : dict of str -> float, optional
        Per-term multipliers; names must be among ``terms`` (unlisted terms get
        1). All equal by default.
    rescale : bool, optional
        Min-max each term first (default True).

    Returns
    -------
    pandas.DataFrame
        One column per (rescaled) term, an ``index`` column (the weighted-mean
        priority on ``[0, 1]``) and an integer ``rank`` (1 = highest; ties take
        the min rank; NaN index keeps NaN rank). ``df.attrs["weights"]`` holds
        the weights used.
    """
    terms = dict(terms)
    if not terms:
        raise ValueError("supply at least one named term vector (e.g. Dg=..., A=...)")
    lengths = {len(np.asarray(v)) for v in terms.values()}
    if len(lengths) != 1:
        raise ValueError("all terms must have the same length (one value per patch)")

    nm = list(terms)
    M = np.column_stack([np.asarray(terms[k], dtype=float) for k in nm])
    if rescale:
        M = np.column_stack([_rescale01(M[:, c]) for c in range(M.shape[1])])

    w = np.ones(len(nm))
    if weights is not None:
        bad = set(weights) - set(nm)
        if bad:
            raise ValueError(f"weights for terms not supplied: {', '.join(sorted(bad))}")
        for i, k in enumerate(nm):
            if k in weights:
                w[i] = float(weights[k])

    # weighted mean over the terms present per patch (renormalise where NaN)
    W = np.broadcast_to(w, M.shape).copy()
    W[np.isnan(M)] = np.nan
    den = np.nansum(W, axis=1)
    with np.errstate(invalid="ignore"):
        index = np.nansum(W * M, axis=1) / den
    index[den == 0] = np.nan

    out = pd.DataFrame(M, columns=nm)
    out["index"] = index
    # rank(-index, ties="min", na_last="keep"): 1 = highest index
    out["rank"] = pd.Series(-index).rank(method="min").astype("Int64")
    out.attrs["weights"] = dict(zip(nm, w))
    return out


def rank_calculate(
    patches,
    network=None,
    origin_xy=None,
    *,
    extra_terms=None,
    areas=None,
    area_col="acres",
    prob_col="prob_mean",
    weights=None,
    target_area=50.0,
    cap_q=0.95,
    speeds=None,
    offtrail_speed=5.0,
    offtrail_penalty=0.5,
    invert=True,
    cluster_eps_km=None,
    cluster_buffer_m=1500.0,
):
    """Score patches from the operational terms, mirroring R's ``rank_calculate``.

    Terms are included only when their inputs are present, so the ranking
    degrades gracefully. Accessibility (A) and complementarity (C) are computed
    from ``network`` + ``origin_xy``; the patch suitability term (P) is read from
    ``prob_col``; and the raster-derived terms (Dg, Dr, S) are taken from
    ``extra_terms`` (their surface computation is out of scope here).

    Parameters
    ----------
    patches : geopandas.GeoDataFrame
        Units to score (points or polygons), CRS in metres.
    network : geopandas.GeoDataFrame, optional
        Road/trail lines for A/C (see :func:`lsed.network.build_network`).
    origin_xy : array_like, optional
        Trip origin(s) ``(x, y)`` for A; passed through to accessibility.
    extra_terms : dict of str -> array_like, optional
        Precomputed per-patch terms (e.g. ``{"Dg": ..., "Dr": ..., "S": ...}``),
        one value per patch in ``patches`` order.
    areas : array_like, optional
        Patch areas (acres) for C; defaults to ``patches[area_col]``.
    area_col, prob_col : str, optional
        Column names for the area (C) and suitability (P) terms.
    weights : dict, optional
        Per-term multipliers for :func:`rank_index`.
    target_area, cap_q : float, optional
        Complementarity parameters.
    speeds : dict, optional
        Per-``highway`` speed lookup (default :data:`lsed.network.osm_speeds`).
    offtrail_speed, offtrail_penalty : float, optional
        Accessibility off-trail hop parameters.
    invert : bool, optional
        Passed to the A/C term functions.

    Returns
    -------
    geopandas.GeoDataFrame
        ``patches`` with the rescaled term columns, ``index`` and ``rank``
        appended.
    """
    speeds = osm_speeds if speeds is None else speeds
    terms: dict[str, np.ndarray] = {}

    if extra_terms:
        for key, vec in extra_terms.items():
            vec = np.asarray(vec, dtype=float)
            if len(vec) != len(patches):
                raise ValueError(f"extra_terms['{key}'] must have one value per patch")
            terms[key] = vec

    # A, C over the network.
    if network is not None and origin_xy is not None:
        if areas is None:
            if area_col not in patches:
                raise ValueError(f"areas needed for C: pass areas= or a '{area_col}' column")
            areas = patches[area_col].to_numpy()
        areas = np.asarray(areas, dtype=float)
        block_kw = dict(
            speeds=speeds, offtrail_speed=offtrail_speed,
            offtrail_penalty=offtrail_penalty, target_area=target_area,
            cap_q=cap_q, invert=invert,
        )
        n = len(patches)
        acc_hr = np.full(n, np.nan)
        asm_hr = np.full(n, np.nan)
        if cluster_eps_km is None:
            acc_hr, asm_hr = _ac_block(network, patches, origin_xy, areas, **block_kw)
        else:
            # per-cluster: each cluster routes on all roads (the drive-in) plus
            # only the trails near its own bbox, so work scales with cluster size.
            from shapely.geometry import box as _box

            cl = cluster_patches(patches, eps_km=cluster_eps_km)
            roads = network[network["class"] == "road"]
            trails = network[network["class"] == "trail"]
            for gid in np.unique(cl):
                idx = np.where(cl == gid)[0]
                sub = patches.iloc[idx]
                bb = _box(*sub.total_bounds).buffer(cluster_buffer_m)
                local = trails[trails.intersects(bb)]
                net_lines = pd.concat([roads, local])
                a, c = _ac_block(
                    net_lines, sub, origin_xy, areas[idx], **block_kw
                )
                acc_hr[idx] = a
                asm_hr[idx] = c
        # raw negative hours; rank_index rescales so low hours -> high priority
        terms["A"] = -acc_hr
        terms["C"] = -asm_hr

    # P - model suitability of the patch itself.
    if prob_col is not None and prob_col in patches:
        terms["P"] = patches[prob_col].to_numpy(dtype=float)

    if not terms:
        raise ValueError("no terms available: supply a network+origin, extra_terms, or a prob_col")

    scored = rank_index(terms, weights=weights)
    out = patches.copy()
    for col in scored.columns:
        out[col] = scored[col].to_numpy()
    out.attrs["weights"] = scored.attrs["weights"]
    return out
