import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import Point

from lsed import (
    aggregate,
    cellsize,
    rank_calculate,
    rasterize_mask,
    raster_terms,
    read_grid,
    sample,
    write_grid,
)

CRS = "EPSG:32613"
# 10 m cells, origin at (0, 100) with y decreasing downward (north-up)
TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)


def test_cellsize_from_transform():
    assert cellsize(TRANSFORM) == 10.0


def test_sample_reads_the_right_cell_and_flags_offgrid():
    grid = np.arange(100, dtype=float).reshape(10, 10)
    # cell (row=0, col=3) centre is at x=35, y=95
    vals = sample(grid, TRANSFORM, [(35, 95), (-50, 50)])
    assert vals[0] == grid[0, 3]
    assert np.isnan(vals[1])  # off the grid -> NaN


def test_geotiff_roundtrip(tmp_path):
    grid = np.random.default_rng(0).random((8, 12)).astype(float)
    path = tmp_path / "surface.tif"
    write_grid(path, grid, TRANSFORM, CRS)
    back, tr, crs = read_grid(path)
    assert back.shape == grid.shape
    assert np.allclose(back, grid, atol=1e-6)
    assert cellsize(tr) == 10.0


def test_rasterize_mask_burns_points():
    mask = rasterize_mask([Point(35, 95), Point(5, 5)], (10, 10), TRANSFORM)
    assert mask.dtype == bool
    assert mask[0, 3]  # (x=35, y=95) -> row 0, col 3
    assert mask.sum() == 2


def test_aggregate_halves_resolution_and_scales_transform():
    grid = np.ones((4, 4))
    out, tr = aggregate(grid, TRANSFORM, factor=2)
    assert out.shape == (2, 2)
    assert np.allclose(out, 1.0)  # mean of ones
    assert cellsize(tr) == 20.0  # pixel doubled
    assert (tr.c, tr.f) == (TRANSFORM.c, TRANSFORM.f)  # origin unchanged


def test_aggregate_handles_partial_edge_blocks():
    grid = np.ones((3, 3))
    out, _ = aggregate(grid, TRANSFORM, factor=2)
    assert out.shape == (2, 2)
    assert np.allclose(out, 1.0)  # NaN-padded edges ignored, not zero-filled


def test_raster_terms_feed_rank_calculate_end_to_end():
    # a 20x20 suitability surface, higher toward the NE; one known population
    rows, cols = np.mgrid[0:20, 0:20]
    suit = (cols - rows + 20) / 40.0  # in [0, 1]
    occ = gpd.GeoDataFrame(geometry=[Point(5, 95)], crs=CRS)  # near top-left
    patches = gpd.GeoDataFrame(
        {"prob_mean": [0.8, 0.3]},
        geometry=[Point(20, 80), Point(180, 20)],
        crs=CRS,
    )
    terms = raster_terms(patches, suit, TRANSFORM, occ, agg=2)
    assert set(terms) == {"Dg", "Dr"}
    assert all(len(v) == len(patches) for v in terms.values())
    # patch near the occurrence should score higher on Dg (inverted distance)
    assert terms["Dg"][0] > terms["Dg"][1]

    out = rank_calculate(patches, extra_terms=terms)
    for col in ("Dg", "Dr", "P", "index", "rank"):
        assert col in out.columns
    assert out["index"].between(0, 1).all()
