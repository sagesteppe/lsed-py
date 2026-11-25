"""Network contraction: collapse junction-to-junction runs.

A Python mirror of the topology half of the R package's ``contract_network``
(``fn_network.R``). The R version leans on ``sfnetworks::to_spatial_smooth``;
this reimplements the degree-2 within-class smoothing directly over an edge list
so no ``sfnetworks`` equivalent is required.

The companion *subdivision* step (``to_spatial_subdivision``: split an edge
where another feature's vertex meets its interior) needs real geometry and is
not included; for pre-noded OSM input, where junctions already share
coordinates, it is a no-op. Pass already-noded edges.
"""

from __future__ import annotations

__all__ = ["smooth_degree2"]


def smooth_degree2(edges, require_equal="hw", keep=None):
    """Collapse degree-2 same-class chains into single edges.

    Every node of degree 2 whose two incident edges agree on
    ``require_equal`` is removed and its two edges merged into one, summing
    length (distance-preserving) and keeping the class. Junctions (degree != 2),
    class-boundary pseudo-nodes, and merges that would form a self-loop are left
    intact, matching ``sfnetworks::to_spatial_smooth``.

    Parameters
    ----------
    edges : list of dict
        Each edge is ``{"u": node, "v": node, "length": float, <require_equal>: value}``;
        ``u``/``v`` are any hashable node ids, treated as undirected.
    require_equal : str, optional
        Edge attribute two edges must share to be smoothed together
        (default ``"hw"`` -- never merge across classes).
    keep : iterable, optional
        Node ids never to smooth (e.g. origins).

    Returns
    -------
    list of dict
        A new edge list; surviving junction node ids are unchanged.
    """
    keep = set() if keep is None else set(keep)
    edges = [dict(e) for e in edges]  # copy; we mutate
    live = set(range(len(edges)))
    inc: dict[object, set[int]] = {}
    for i, e in enumerate(edges):
        inc.setdefault(e["u"], set()).add(i)
        inc.setdefault(e["v"], set()).add(i)

    work = [nd for nd, es in inc.items() if len(es) == 2]
    nxt = len(edges)

    while work:
        nd = work.pop()
        if nd in keep:
            continue
        es = inc.get(nd, set())
        if len(es) != 2:
            continue  # degree changed / already gone
        i, j = tuple(es)
        ei, ej = edges[i], edges[j]
        if ei[require_equal] != ej[require_equal]:
            continue  # class boundary -> keep pseudo-node
        a = ei["v"] if ei["u"] == nd else ei["u"]  # far end of edge i
        b = ej["v"] if ej["u"] == nd else ej["u"]  # far end of edge j
        if a == b:
            continue  # would form a self-loop -> keep
        merged = {
            "u": a,
            "v": b,
            "length": ei["length"] + ej["length"],
            require_equal: ei[require_equal],
        }
        k = nxt
        nxt += 1
        edges.append(merged)
        live.add(k)
        for node in (ei["u"], ei["v"]):
            inc[node].discard(i)
        for node in (ej["u"], ej["v"]):
            inc[node].discard(j)
        live.discard(i)
        live.discard(j)
        inc.pop(nd, None)
        inc.setdefault(a, set()).add(k)
        inc.setdefault(b, set()).add(k)
        for node in (a, b):
            if len(inc.get(node, ())) == 2 and node not in keep:
                work.append(node)

    return [edges[i] for i in sorted(live)]
