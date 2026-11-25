import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString, Point

from lsed import (
    cluster_patches,
    guess_record_type,
    rank_calculate,
    survey_field,
    survey_raster,
)

CRS = "EPSG:32613"


def test_cluster_patches_groups_by_distance():
    # two tight groups ~10 km apart; eps 5 km keeps them separate
    pts = [Point(0, 0), Point(100, 0), Point(10_000, 0), Point(10_100, 0)]
    patches = gpd.GeoDataFrame(geometry=pts, crs=CRS)
    lab = cluster_patches(patches, eps_km=5)
    assert lab[0] == lab[1] and lab[2] == lab[3]
    assert lab[0] != lab[2]  # two clusters


def test_cluster_patches_merges_when_eps_large():
    pts = [Point(0, 0), Point(100, 0), Point(10_000, 0), Point(10_100, 0)]
    patches = gpd.GeoDataFrame(geometry=pts, crs=CRS)
    assert len(set(cluster_patches(patches, eps_km=50))) == 1  # all one cluster


def test_cluster_patches_accepts_coord_array():
    xy = np.array([[0, 0], [100, 0], [9000, 0]])
    lab = cluster_patches(xy, eps_km=5)
    assert lab[0] == lab[1] and lab[0] != lab[2]


def test_guess_record_type_matches_plot_sources():
    out = guess_record_type(
        ["FIA plot data", "iNaturalist obs", "AIM terrestrial", None, "random photo"]
    )
    assert out == ["Plot", "Opportunistic", "Plot", None, "Opportunistic"]


def test_survey_raster_shape_and_bounds():
    field = survey_field(
        np.array([[55.0, 45.0]]), ["A"], sigma=20.0, record_type=["Plot"]
    )
    tr = Affine(10, 0, 0, 0, -10, 100)  # 10x10 grid; record sits at cell (5,5) centre
    s = survey_raster(field, tr, (10, 10), rescale="max")
    assert s.shape == (10, 10)
    assert np.nanmax(s) <= 1.0  # rescaled by max
    assert np.unravel_index(np.nanargmax(s), s.shape) == (5, 5)


def test_rank_calculate_cluster_path_matches_single_for_one_cluster():
    # one tight cluster -> clustered and single-graph routes should agree
    network = gpd.GeoDataFrame(
        {
            "class": ["road", "trail"],
            "highway": ["residential", "path"],
            "geometry": [
                LineString([(0, 0), (150, 0)]),
                LineString([(150, 0), (190, 40)]),
            ],
        },
        crs=CRS,
    )
    patches = gpd.GeoDataFrame(
        {"acres": [8.0, 8.0, 8.0]},
        geometry=[Point(60, 30), Point(120, 20), Point(90, 15)],
        crs=CRS,
    )
    single = rank_calculate(patches, network=network, origin_xy=[(0, 0)])
    clustered = rank_calculate(
        patches, network=network, origin_xy=[(0, 0)],
        cluster_eps_km=50, cluster_buffer_m=2000,
    )
    # same single cluster + generous buffer -> identical accessibility hours
    assert np.allclose(single["A"].to_numpy(), clustered["A"].to_numpy(), equal_nan=True)


def test_rank_calculate_cluster_path_runs_for_separate_clusters():
    network = gpd.GeoDataFrame(
        {
            "class": ["road", "trail", "trail"],
            "highway": ["residential", "path", "path"],
            "geometry": [
                LineString([(0, 0), (20_000, 0)]),  # long road spanning both
                LineString([(100, 0), (140, 40)]),
                LineString([(10_100, 0), (10_140, 40)]),
            ],
        },
        crs=CRS,
    )
    patches = gpd.GeoDataFrame(
        {"acres": [8.0, 8.0]},
        geometry=[Point(120, 20), Point(10_120, 20)],  # ~10 km apart
        crs=CRS,
    )
    out = rank_calculate(
        patches, network=network, origin_xy=[(0, 0)], cluster_eps_km=5
    )
    assert "A" in out.columns and len(out) == 2
    assert out["A"].notna().all()  # both clusters routed
