"""Road/trail travel-time network: accessibility (A) and complementarity (C).

A Python mirror of the operational-term engine in the R package's
``fn_network.R``. Travel runs on the linestrings -- roads and trails as a
weighted graph, each edge timed by its class speed -- and only the last hop from
the network to a patch centroid drops to an off-trail cost. Because patches
become graph nodes, the same network answers both terms.

sf/sfnetworks/igraph are replaced by shapely/geopandas for geometry and
scipy.sparse.csgraph for shortest paths. The optional DEM slope term (R's
``terra`` extract) is omitted; :func:`hop_speed` takes pre-sampled elevations.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import STRtree
from shapely.ops import nearest_points

__all__ = [
    "osm_speeds",
    "rescale01",
    "hop_speed",
    "build_network",
    "accessibility",
    "complementarity",
]

#: Default per-``highway`` travel speeds (km/h), keyed on the OSM ``highway``
#: tag; an unlisted value falls back to the road/trail class speed.
osm_speeds = {
    "motorway": 90,
    "trunk": 80,
    "primary": 70,
    "secondary": 55,
    "tertiary": 45,
    "unclassified": 35,
    "residential": 30,
    "service": 20,
    "living_street": 15,
    "track": 15,
    "road": 30,
    "path": 4,
    "footway": 4,
    "bridleway": 4,
    "cycleway": 10,
    "steps": 2,
}


def rescale01(x, invert=False):
    """Min-max rescale to ``[0, 1]`` over the finite values.

    A degenerate range (all equal, or no finite values) maps to all zeros.
    With ``invert=True`` the result is ``1 - rescaled`` (the index form, where
    low cost becomes a high score).
    """
    x = np.asarray(x, dtype=float)
    fin = x[np.isfinite(x)]
    if fin.size == 0:
        return x * 0.0
    lo, hi = fin.min(), fin.max()
    out = (x - lo) / (hi - lo) if hi > lo else x * 0.0
    return 1.0 - out if invert else out


def hop_speed(run, z1=None, z2=None, offtrail_speed=5.0, penalty=0.5):
    """Off-trail Tobler speed (km/h) per hop, times the off-trail penalty.

    With elevations, slope is ``|z2 - z1| / run`` from the centroid to its
    network entry point; without them, a flat ``offtrail_speed``. The DEM
    extract itself (R's ``terra::extract``) is a raster concern kept outside
    this function: pass pre-sampled ``z1`` / ``z2`` for the slope term.

    Parameters
    ----------
    run : array_like
        Hop length(s) in metres.
    z1, z2 : array_like, optional
        Centroid and network-node elevations; both required for the slope term.
    offtrail_speed : float, optional
        Flat-ground off-trail speed (km/h) when no elevations are given.
    penalty : float, optional
        Multiplier on the hop speed (default 0.5); 1 removes the penalty.

    Returns
    -------
    numpy.ndarray
        Hop speed in km/h.
    """
    run = np.asarray(run, dtype=float)
    if z1 is None or z2 is None:
        return np.full(run.shape, offtrail_speed * penalty)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    slope = np.where(run > 0, np.abs(z2 - z1) / run, 0.0)
    spd = 6.0 * np.exp(-3.5 * np.abs(slope + 0.05))  # Tobler km/h
    spd = np.where(~np.isfinite(spd) | (spd <= 0), offtrail_speed, spd)
    return spd * penalty


def build_network(gdf, road_speed=30, trail_speed=4, snap=1, connect=50, speeds=None):
    """Build a travel-time graph from road/trail linestrings.

    Nodes are shared (snapped) endpoints; each edge is timed as
    ``(length_km / speed)`` hours from its ``highway`` speed (class fallback).
    Degree-1 stub ends of one class are bridged to the nearest node of the other
    class within ``connect`` metres, closing trailhead gaps.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        LINESTRINGs in a projected CRS (metres) with a ``class`` ("road"/"trail")
        and optional ``highway`` column.
    road_speed, trail_speed : float, optional
        Fallback class speeds (km/h) for a ``highway`` not in ``speeds``.
    snap : float, optional
        Grid (metres) that endpoint coordinates snap to when forming nodes.
    connect : float, optional
        Tolerance (metres) for bridging a class-crossing stub end; 0 disables.
    speeds : dict, optional
        Per-``highway`` speed lookup (default :data:`osm_speeds`).

    Returns
    -------
    dict
        ``n_nodes``, ``nodes_xy`` (array), ``edges`` (list of dicts with
        ``u, v, cls, hw, len, speed, weight, geom``) and the ``edge_u``,
        ``edge_v``, ``edge_w`` arrays for the scipy graph.
    """
    speeds = osm_speeds if speeds is None else speeds

    node_id: dict[tuple, int] = {}
    node_xy: list[tuple] = []

    def node_of(pt):
        k = (round(pt[0] / snap) * snap, round(pt[1] / snap) * snap)
        if k not in node_id:
            node_id[k] = len(node_xy)
            node_xy.append(k)
        return node_id[k]

    cls = (
        gdf["class"].astype(str).to_numpy()
        if "class" in gdf
        else np.array(["trail"] * len(gdf))
    )
    hw = (
        gdf["highway"].astype(object).to_numpy()
        if "highway" in gdf
        else np.array([None] * len(gdf), dtype=object)
    )

    edges = []
    for geom, c, h in zip(gdf.geometry, cls, hw):
        coords = list(geom.coords)
        u = node_of(coords[0])
        v = node_of(coords[-1])
        spd = speeds.get(h) if h is not None else None
        if spd is None:
            spd = road_speed if c == "road" else trail_speed
        length = geom.length
        w = (length / 1000.0) / spd
        if u != v and np.isfinite(w) and w > 0:
            edges.append(
                dict(u=u, v=v, cls=c, hw=h, len=length, speed=spd, weight=w, geom=geom)
            )

    nodes_xy = np.array(node_xy, dtype=float)

    if connect > 0 and edges:
        deg = np.zeros(len(nodes_xy), dtype=int)
        for e in edges:
            deg[e["u"]] += 1
            deg[e["v"]] += 1
        stubs = set(np.where(deg == 1)[0])

        def class_nodes(cname):
            s = set()
            for e in edges:
                if e["cls"] == cname:
                    s.update((e["u"], e["v"]))
            return s

        road_nodes = class_nodes("road")
        trail_nodes = class_nodes("trail")
        extra = []

        def bridge(src_class_nodes, dst_nodes):
            src = sorted(stubs & src_class_nodes)
            dst = sorted(dst_nodes)
            if not src or not dst:
                return
            dst_xy = nodes_xy[dst]
            for s in src:
                d = np.hypot(*(nodes_xy[s] - dst_xy).T)
                j = int(np.argmin(d))
                gap = float(d[j])
                tgt = dst[j]
                if gap <= connect and s != tgt:
                    extra.append(
                        dict(
                            u=s,
                            v=tgt,
                            weight=(gap / 1000.0) / trail_speed,
                            cls="connect",
                            hw=None,
                            len=gap,
                            speed=trail_speed,
                            geom=None,
                        )
                    )

        bridge(trail_nodes, road_nodes)
        bridge(road_nodes, trail_nodes)
        seen = set()
        for e in extra:
            k = (min(e["u"], e["v"]), max(e["u"], e["v"]))
            if k not in seen:
                seen.add(k)
                edges.append(e)

    eu = np.array([e["u"] for e in edges], dtype=int)
    ev = np.array([e["v"] for e in edges], dtype=int)
    ew = np.array([e["weight"] for e in edges], dtype=float)
    return dict(
        n_nodes=len(nodes_xy),
        nodes_xy=nodes_xy,
        edges=edges,
        edge_u=eu,
        edge_v=ev,
        edge_w=ew,
    )


def accessibility(
    net, patches, origins_xy=None, offtrail_speed=5.0, offtrail_penalty=0.5, invert=True
):
    """Accessibility of patches over the network (term A).

    Drive the roads from an origin, walk the trails, then an off-trail hop to
    each patch centroid; ``access_hr`` is the least-cost origin -> network -> hop
    in hours. Each patch enters through the nearest reachable edge, projected to
    the true nearest point on that edge (not a distant junction). The returned
    augmented graph carries the patches as nodes, ready for
    :func:`complementarity`.

    Parameters
    ----------
    net : dict
        A :func:`build_network` result (edges must carry ``geom``).
    patches : geopandas.GeoDataFrame
        Points or polygons; polygons are reduced to centroids.
    origins_xy : array_like, optional
        ``(x, y)`` trip starts, snapped to the nearest road node; ``None``
        (default) treats every road node as a zero-cost origin.
    offtrail_speed, offtrail_penalty : float, optional
        Flat off-trail hop speed (km/h) and its penalty multiplier.
    invert : bool, optional
        If True (default) ``A`` is the index form ``1 - rescaled(access_hr)``.

    Returns
    -------
    dict
        ``access_hr`` (hours, NaN if unreachable), ``A`` (rescaled score),
        ``net`` (augmented graph with ``patch_idx``) and ``patch_idx``.
    """
    nxy = net["nodes_xy"]
    edges = net["edges"]
    n0 = net["n_nodes"]

    g = csr_matrix(
        (net["edge_w"], (net["edge_u"], net["edge_v"])), shape=(n0, n0)
    )

    road_nodes = sorted(
        {e["u"] for e in edges if e["cls"] == "road"}
        | {e["v"] for e in edges if e["cls"] == "road"}
    )
    if not road_nodes:
        road_nodes = list(range(n0))

    if origins_xy is None:
        onodes = road_nodes
    else:
        rxy = nxy[road_nodes]
        onodes = sorted(
            {
                road_nodes[int(np.argmin(np.hypot(*(np.asarray(o) - rxy).T)))]
                for o in origins_xy
            }
        )

    d0 = dijkstra(g, directed=False, indices=onodes).min(axis=0)
    reach = set(np.where(np.isfinite(d0))[0])
    if not reach:
        raise ValueError("no network nodes reachable from origin(s)")

    is_poly = patches.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any()
    cen = list(patches.geometry.centroid if is_poly else patches.geometry)
    pc = np.array([(p.x, p.y) for p in cen])

    usable = [
        i
        for i, e in enumerate(edges)
        if (e["u"] in reach or e["v"] in reach) and e["u"] != e["v"] and e["geom"] is not None
    ]
    if not usable:
        raise ValueError("no network edge reachable from origin(s)")
    tree = STRtree([edges[i]["geom"] for i in usable])

    acc = np.full(len(cen), np.inf)
    pu, pv, pw = [], [], []
    for p, c in enumerate(cen):
        ei = usable[int(tree.query_nearest(c)[0])]
        e = edges[ei]
        geom = e["geom"]
        proj_pt = nearest_points(c, geom)[1]
        hop_m = c.distance(geom)
        offA = min(max(geom.project(proj_pt), 0.0), e["len"])
        offB = e["len"] - offA
        hop_hr = hop_m / 1000.0 / (offtrail_speed * offtrail_penalty)
        wA = hop_hr + (offA / 1000.0) / e["speed"]
        wB = hop_hr + (offB / 1000.0) / e["speed"]
        dA = d0[e["u"]] if np.isfinite(d0[e["u"]]) else np.inf
        dB = d0[e["v"]] if np.isfinite(d0[e["v"]]) else np.inf
        acc[p] = min(dA + wA, dB + wB)
        pid = n0 + p
        pu += [pid, pid]
        pv += [e["u"], e["v"]]
        pw += [wA, wB]
    acc[~np.isfinite(acc)] = np.nan

    aug = dict(
        n_nodes=n0 + len(cen),
        nodes_xy=np.vstack([nxy, pc]) if len(cen) else nxy,
        edge_u=np.concatenate([net["edge_u"], np.array(pu, dtype=int)]),
        edge_v=np.concatenate([net["edge_v"], np.array(pv, dtype=int)]),
        edge_w=np.concatenate([net["edge_w"], np.array(pw, dtype=float)]),
        patch_idx=np.arange(n0, n0 + len(cen)),
    )
    return dict(
        access_hr=acc, A=rescale01(acc, invert), net=aug, patch_idx=aug["patch_idx"]
    )


def complementarity(aug, areas, target_area=50.0, cap_q=0.95, invert=True):
    """Complementarity of patches: can a day's work be assembled nearby? (C)

    An area-weighted isolation penalty. From each patch, walk the network
    cheapest-first accumulating neighbouring patch acreage until ``target_area``
    is gathered; the travel cost of that bundle is the isolation measure. A dense
    cluster assembles cheaply; a lone fragment must range far (or cannot gather
    it at all). The score is flipped so well-connected patches are high.

    Parameters
    ----------
    aug : dict
        The augmented ``net`` from :func:`accessibility` (patches as nodes),
        carrying ``n_nodes``, ``edge_u/v/w`` and ``patch_idx``.
    areas : array_like
        Patch areas (acres), one per ``patch_idx`` in order.
    target_area : float, optional
        A day's worth of patch area to assemble (default 50).
    cap_q : float or None, optional
        Upper quantile at which the raw cost's long right tail (and unreachable
        ``inf``) is clamped before rescaling (default 0.95); ``None`` disables.
    invert : bool, optional
        If True (default) ``C`` is the index form ``1 - rescaled(cost)``.

    Returns
    -------
    dict
        ``assemble_hr`` (travel hours, clamped) and ``C`` on ``[0, 1]``, in
        ``patch_idx`` order.
    """
    patch_idx = np.asarray(aug["patch_idx"])
    areas = np.asarray(areas, dtype=float)
    if len(areas) != len(patch_idx):
        raise ValueError("areas must align with patch_idx")
    if cap_q is not None and not (0 < cap_q <= 1):
        raise ValueError("cap_q must be in (0, 1]")

    n_nodes = aug["n_nodes"]
    g = csr_matrix(
        (aug["edge_w"], (aug["edge_u"], aug["edge_v"])), shape=(n_nodes, n_nodes)
    )
    D = dijkstra(g, directed=False, indices=patch_idx)[:, patch_idx]

    n = len(patch_idx)
    cost = np.empty(n)
    for i in range(n):
        d = D[i]
        order = np.argsort(d, kind="stable")  # self (d=0) first
        cum = np.cumsum(areas[order])
        if cum[-1] < target_area:
            cost[i] = np.inf
        else:
            cost[i] = d[order[int(np.argmax(cum >= target_area))]]

    fin = np.isfinite(cost)
    if cap_q is not None and fin.any():
        cap = np.quantile(cost[fin], cap_q)
        cost[cost > cap] = cap  # finite outliers and inf alike -> cap
        fin = np.isfinite(cost)
    cost[~fin] = cost[fin].max() if fin.any() else 1.0

    return dict(assemble_hr=cost, C=rescale01(cost, invert))
