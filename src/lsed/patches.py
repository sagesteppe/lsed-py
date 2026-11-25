"""Raster-graph helpers for patch / subpatch delineation.

A Python mirror of the raster-graph pieces of the R package's ``fn_patches.R``:
rook adjacency over a cell set, connected-component labelling, a band-priority
union-find merge of sub-threshold pieces, and a geometric k-means size cap.

All indices here are 0-based (the R originals are 1-based); a ``band`` value
indexes ``minac`` directly, so an R band ``1..B`` becomes ``0..B-1`` here.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = ["rook_edges", "comp_labels", "snap_pieces", "kcap", "cluster_patches"]


def rook_edges(gc, nc, nr):
    """East/south rook adjacency among a set of global cell indices.

    Returns the adjacency as *local* index pairs (positions within ``gc``).
    Only east and south neighbours are emitted, so each undirected rook edge is
    produced once, from its lower-index endpoint.

    Parameters
    ----------
    gc : array_like
        Global (0-based) raster cell indices, row-major.
    nc, nr : int
        Column and row counts of the parent raster.

    Returns
    -------
    tuple of numpy.ndarray
        ``(i, j)`` local index pairs of adjacent cells.
    """
    gc = np.asarray(gc)
    pos = {int(c): k for k, c in enumerate(gc)}  # global cell -> local index
    r = gc // nc
    cc = gc % nc
    out_i, out_j = [], []
    for shift, mask in ((1, cc < nc - 1), (nc, r < nr - 1)):
        for k in np.where(mask)[0]:
            nb = pos.get(int(gc[k]) + shift)
            if nb is not None:
                out_i.append(int(k))
                out_j.append(nb)
    return np.array(out_i, dtype=int), np.array(out_j, dtype=int)


def comp_labels(n, ei, ej):
    """Connected-component labels for ``n`` nodes given an undirected edge set.

    Labels are remapped to first-appearance order (node 0's component is 0, the
    next new component 1, ...) so the output is deterministic and independent of
    the traversal order, matching the R ``comp_labels`` contract.

    Parameters
    ----------
    n : int
        Number of nodes.
    ei, ej : array_like
        Equal-length endpoint arrays of each edge.

    Returns
    -------
    numpy.ndarray
        Component label per node, values ``0..m-1``.
    """
    ei = np.asarray(ei)
    ej = np.asarray(ej)
    g = coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n))
    _, m = connected_components(g, directed=False)
    seen: dict[int, int] = {}
    out = np.empty(n, dtype=int)
    for i, v in enumerate(m):
        v = int(v)
        if v not in seen:
            seen[v] = len(seen)
        out[i] = seen[v]
    return out


def snap_pieces(piece, ei, ej, area, band, minac, npass=3):
    """Snap sub-threshold band-pieces into their best contiguous neighbour.

    Sequential snap/merge over a region-adjacency graph: repeatedly absorb each
    piece whose accumulated area is below its band's floor (``minac[band]``)
    into a contiguous neighbour, preferring the highest band and then the
    largest area. Passes repeat until one makes no merges (or ``npass`` is hit),
    letting chains collapse.

    Parameters
    ----------
    piece : array_like
        Per-cell band-piece labels ``0..np-1`` (e.g. from :func:`comp_labels`).
    ei, ej : array_like
        *All* rook edges as local cell-index pairs (from :func:`rook_edges`);
        cross-piece edges define piece adjacency.
    area : array_like
        Per-piece area (length ``np``).
    band : array_like
        Per-piece band (length ``np``), low to high; indexes ``minac``.
    minac : array_like
        Per-band minimum size; a piece below ``minac[band]`` is a merge
        candidate.
    npass : int, optional
        Maximum number of snap passes (default 3).

    Returns
    -------
    numpy.ndarray
        The resolved root piece for each piece; map per cell with ``root[piece]``.
    """
    piece = np.asarray(piece)
    ei = np.asarray(ei)
    ej = np.asarray(ej)
    arr = np.asarray(area, dtype=float).copy()  # accumulates
    brr = np.asarray(band)  # never mutated
    minac = np.asarray(minac, dtype=float)
    n = len(area)

    # region-adjacency: a dict-of-sets dedupes shared borders for free
    dif = piece[ei] != piece[ej]
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in zip(piece[ei[dif]], piece[ej[dif]]):
        adj[a].add(int(b))
        adj[b].add(int(a))

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = int(parent[x])
        return x

    for _ in range(npass):
        did = 0
        for p in np.argsort(arr, kind="stable"):  # order(arr), once per pass
            p = int(p)
            rp = find(p)
            if arr[rp] >= minac[brr[rp]]:
                continue
            rr = {find(nbr) for nbr in adj[p]}
            rr.discard(rp)
            if not rr:
                continue
            rr = np.fromiter(rr, dtype=int)
            # best neighbour: highest band, then largest area (both descending)
            tgt = int(rr[np.lexsort((-arr[rr], -brr[rr]))][0])
            parent[rp] = tgt
            arr[tgt] += arr[rp]
            did += 1
        if did == 0:
            break

    return np.array([find(p) for p in range(n)])


def kcap(gc, capn, nc, seed=0):
    """Geometric k-means split with a hard per-cluster cell cap.

    Split a cell set into ``ceil(n / capn)`` compact pieces, each at or below
    ``capn`` cells, by k-means on cell ``(col, row)`` positions; ``k`` is bumped
    until no cluster exceeds the cap. Sets at or under the cap pass through as a
    single piece.

    Unlike the deterministic helpers above, this does not reproduce the R
    partition cell-for-cell: it uses ``scipy.cluster.vq.kmeans2`` (k-means++)
    rather than R's Hartigan-Wong, so the exact assignment can differ. The cap
    invariant (no cluster larger than ``capn``) always holds.

    Parameters
    ----------
    gc : array_like
        Global (0-based) raster cell indices.
    capn : int
        Maximum piece size in cells.
    nc : int
        Column count of the parent raster (to recover col/row from ``gc``).
    seed : int, optional
        RNG seed for the k-means start (stands in for R's ``set.seed``).

    Returns
    -------
    numpy.ndarray
        Cluster label per cell (all zeros when ``n <= capn``).
    """
    gc = np.asarray(gc)
    n = len(gc)
    if n <= capn:
        return np.zeros(n, dtype=int)
    xy = np.column_stack((gc % nc, gc // nc)).astype(float)
    k = int(np.ceil(n / capn))
    while True:
        _, lab = kmeans2(xy, k, minit="++", seed=seed)
        if np.bincount(lab, minlength=k).max() <= capn:
            break
        k += 1
    return lab


def cluster_patches(patches, eps_km=5.0):
    """Group patch centroids that lie within ``eps_km`` of one another.

    A within-distance graph over the centroids, labelled by connected component,
    so downstream work (OSM fetch, per-cluster routing) can be done only where
    patches actually are. Mirrors ``cluster_patches`` in ``fn_download.R``.

    Parameters
    ----------
    patches : geopandas.GeoDataFrame or array_like
        Patches (centroids taken from the geometry) or an ``(n, 2)`` array of
        coordinates, in a projected CRS (metres).
    eps_km : float, optional
        Join distance in km: centroids within this share a cluster (default 5).

    Returns
    -------
    numpy.ndarray
        Integer cluster label per patch, in input order.
    """
    if hasattr(patches, "geometry"):
        cen = patches.geometry.centroid
        xy = np.column_stack([cen.x.to_numpy(), cen.y.to_numpy()])
    else:
        xy = np.asarray(patches, dtype=float)[:, :2]
    n = len(xy)
    if n == 0:
        return np.empty(0, dtype=int)
    pairs = cKDTree(xy).query_pairs(r=eps_km * 1000.0, output_type="ndarray")
    if len(pairs):
        g = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
    else:
        g = coo_matrix((n, n))
    _, labels = connected_components(g, directed=False)
    return labels
