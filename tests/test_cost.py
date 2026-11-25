import numpy as np

from lsed import (
    build_resistance,
    cost_distance,
    geo_distance,
    rescale01_grid,
    tobler_friction,
)


def test_rescale01_grid_invert_and_degenerate():
    g = np.array([[0.0, 1.0], [2.0, 4.0]])
    assert rescale01_grid(g).max() == 1.0 and rescale01_grid(g).min() == 0.0
    assert np.allclose(rescale01_grid(g) + rescale01_grid(g, invert=True), 1.0)
    assert np.all(rescale01_grid(np.full((2, 2), 5.0)) == 0.0)


def test_geo_distance_zero_at_source_and_max_at_corner():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True  # single central source
    d = geo_distance(mask, cellsize=10.0)  # not inverted
    assert d[2, 2] == 0.0  # at the source
    assert d[0, 0] == 1.0  # farthest corner is the rescaled max
    # inverted form flips it
    di = geo_distance(mask, cellsize=10.0, invert=True)
    assert di[2, 2] == 1.0 and di[0, 0] == 0.0


def test_geo_distance_scales_with_cellsize():
    mask = np.zeros((1, 3), dtype=bool)
    mask[0, 0] = True
    # pre-rescale distances are proportional to cellsize, so the rescaled grid is
    # identical; check the transform itself is linear via rescale invariance
    assert np.allclose(geo_distance(mask, 1.0), geo_distance(mask, 100.0))


def test_cost_distance_monotone_from_source_on_uniform_friction():
    fric = np.ones((1, 6))
    src = np.zeros((1, 6), dtype=bool)
    src[0, 0] = True
    cd = cost_distance(fric, src)
    assert cd[0, 0] == 0.0
    assert np.all(np.diff(cd[0]) > 0)  # strictly increasing away from the source


def test_cost_distance_prefers_the_cheap_corridor():
    # a wall of high friction with a low-cost gap; cost to the far side should
    # route through the gap, so the cell behind the gap is cheaper than behind the wall
    fric = np.ones((5, 5)) * 100.0
    fric[2, :] = 100.0
    fric[2, 2] = 1.0  # the gap
    fric[0, :] = 1.0
    fric[4, :] = 1.0
    fric[:, 0] = 1.0
    src = np.zeros((5, 5), dtype=bool)
    src[0, 0] = True
    cd = cost_distance(fric, src, invert=False)
    assert np.isfinite(cd).all()
    assert cd[4, 2] < cd[2, 1]  # the gap route beats punching the wall


def test_cost_distance_rejects_empty_sources():
    try:
        cost_distance(np.ones((3, 3)), np.zeros((3, 3), dtype=bool))
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty source_mask")


def test_tobler_friction_higher_on_slopes():
    flat = np.zeros((5, 5))
    ramp = np.tile(np.arange(5.0) * 20.0, (5, 1))  # 20 m rise per 10 m cell
    assert np.allclose(tobler_friction(flat, 10.0), tobler_friction(flat, 10.0))
    assert tobler_friction(ramp, 10.0).mean() > tobler_friction(flat, 10.0).mean()


def test_build_resistance_weights_clamp_and_integer():
    shape = (3, 3)
    water = np.zeros(shape)
    water[0, 0] = 1
    habitat = np.zeros(shape)  # all-cheap habitat
    res = build_resistance(shape, oceans=water, habitat=habitat, w_ocean=100.0)
    assert res.dtype.kind in "iu"  # integer
    assert res.min() >= 1  # clamped floor
    assert res[0, 0] == 100  # w_ocean * 1 + habitat 0
    assert res[1, 1] == 1  # no layers -> clamped up to the floor of 1
