import numpy as np

from lsed import comp_labels, kcap, rook_edges, snap_pieces


def test_rook_edges_skips_a_removed_cell():
    # 3x3 grid with the centre cell (4) removed
    gc = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    i, j = rook_edges(gc, nc=3, nr=3)
    pairs = {(int(gc[a]), int(gc[b])) for a, b in zip(i, j)}
    assert pairs == {(0, 1), (1, 2), (0, 3), (2, 5), (3, 6), (5, 8), (6, 7), (7, 8)}
    assert not any(4 in p for p in pairs)  # nothing touches the removed cell


def test_comp_labels_first_appearance_order():
    # two components {0,1,2} and {3,4}
    out = comp_labels(5, ei=[0, 1, 3], ej=[1, 2, 4])
    assert list(out) == [0, 0, 0, 1, 1]


def test_snap_pieces_collapses_chain_to_highest_band():
    # pieces 0-1-2-3 in a line; all below floor except the big high-band piece 3
    piece = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    ei = np.array([0, 1, 2, 3, 4, 5, 6])
    ej = np.array([1, 2, 3, 4, 5, 6, 7])
    root = snap_pieces(
        piece, ei, ej,
        area=[1.0, 1.0, 1.0, 10.0],
        band=[0, 1, 2, 3],
        minac=[5.0, 5.0, 5.0, 5.0],
    )
    assert list(root) == [3, 3, 3, 3]  # everything absorbed into piece 3


def test_kcap_passes_through_small_sets():
    assert list(kcap(np.arange(10), capn=30, nc=10)) == [0] * 10


def test_kcap_respects_the_cell_cap():
    lab = kcap(np.arange(100), capn=30, nc=10, seed=1)
    assert np.bincount(lab).max() <= 30
    assert len(set(lab)) >= int(np.ceil(100 / 30))
