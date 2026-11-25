from lsed import smooth_degree2


def _edges(triples):
    return [dict(u=u, v=v, length=length, hw=hw) for u, v, length, hw in triples]


def _as_set(edges):
    return {(e["u"], e["v"], round(e["length"], 2), e["hw"]) for e in edges}


def test_pure_chain_collapses_to_one_edge():
    out = smooth_degree2(_edges([(0, 1, 1, "path"), (1, 2, 1, "path"), (2, 3, 1, "path")]))
    assert _as_set(out) == {(0, 3, 3, "path")}


def test_class_boundary_pseudo_node_survives():
    out = smooth_degree2(_edges([(0, 1, 1, "path"), (1, 2, 1, "road"), (2, 3, 1, "road")]))
    assert _as_set(out) == {(0, 1, 1, "path"), (1, 3, 2, "road")}


def test_junction_node_survives():
    out = smooth_degree2(
        _edges([(0, 1, 1, "road"), (1, 2, 1, "road"), (1, 9, 1, "road"), (2, 3, 1, "road")])
    )
    # node 1 is degree-3 (junction); node 2 smooths away
    assert _as_set(out) == {(0, 1, 1, "road"), (1, 3, 2, "road"), (1, 9, 1, "road")}


def test_triangle_keeps_a_node_to_avoid_self_loop():
    out = smooth_degree2(_edges([(0, 1, 1, "road"), (1, 2, 1, "road"), (2, 0, 1, "road")]))
    # collapses to two nodes joined by a direct (len 1) and a wrap-around (len 2) edge
    lengths = sorted(round(e["length"], 2) for e in out)
    assert len(out) == 2 and lengths == [1, 2]


def test_keep_blocks_smoothing():
    out = smooth_degree2(
        _edges([(0, 1, 1, "path"), (1, 2, 1, "path")]), keep={1}
    )
    assert len(out) == 2  # node 1 protected, nothing merged
