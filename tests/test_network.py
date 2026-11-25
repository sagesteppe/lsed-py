import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from lsed import (
    accessibility,
    build_network,
    complementarity,
    hop_speed,
    osm_speeds,
    rescale01,
)

CRS = "EPSG:32613"


def _road_trail():
    return gpd.GeoDataFrame(
        {
            "class": ["road", "trail"],
            "highway": ["residential", "path"],
            "geometry": [
                LineString([(0, 0), (200, 0)]),
                LineString([(200, 0), (300, 50)]),  # shares node (200, 0)
            ],
        },
        crs=CRS,
    )


def test_rescale01_invert_and_degenerate():
    assert list(rescale01([0.0, 1.0, 2.0])) == [0.0, 0.5, 1.0]
    assert list(rescale01([0.0, 1.0, 2.0], invert=True)) == [1.0, 0.5, 0.0]
    assert list(rescale01([5.0, 5.0])) == [0.0, 0.0]  # no range -> zeros


def test_hop_speed_flat_and_sloped():
    assert np.allclose(hop_speed(np.array([100.0, 200.0])), 2.5)  # 5 * 0.5
    sloped = hop_speed(np.array([100.0, 100.0]), z1=[0, 0], z2=[10, 50])
    assert sloped[0] > sloped[1]  # steeper hop is slower


def test_build_network_nodes_and_speeds():
    net = build_network(_road_trail(), connect=50)
    assert net["n_nodes"] == 3  # shared endpoint at (200, 0)
    road = next(e for e in net["edges"] if e["cls"] == "road")
    assert road["speed"] == osm_speeds["residential"]
    assert np.isclose(road["weight"], (200 / 1000) / 30)


def test_build_network_bridges_a_close_stub():
    gdf = _road_trail()
    gdf = gpd.GeoDataFrame(
        {
            "class": list(gdf["class"]) + ["trail"],
            "highway": list(gdf["highway"]) + ["path"],
            "geometry": list(gdf.geometry) + [LineString([(0, 40), (50, 40)])],
        },
        crs=CRS,
    )
    net = build_network(gdf, connect=50)
    bridges = [e for e in net["edges"] if e["cls"] == "connect"]
    assert len(bridges) == 1  # only the 40 m gap is bridged
    assert np.isclose(bridges[0]["len"], 40.0)


def test_accessibility_prefers_the_patch_near_the_origin():
    net = build_network(_road_trail(), connect=50)
    patches = gpd.GeoDataFrame(
        geometry=[Point(250, 30), Point(100, 20)], crs=CRS
    )
    res = accessibility(net, patches, origins_xy=[(0, 0)])
    # patch near the road/origin is cheaper and scores higher
    assert res["access_hr"][1] < res["access_hr"][0]
    assert res["A"][1] == 1.0 and res["A"][0] == 0.0


def test_complementarity_runs_on_the_augmented_graph():
    net = build_network(_road_trail(), connect=50)
    patches = gpd.GeoDataFrame(
        geometry=[Point(250, 30), Point(260, 35), Point(100, 20)], crs=CRS
    )
    res = accessibility(net, patches, origins_xy=[(0, 0)])
    comp = complementarity(res["net"], areas=[10, 10, 10], target_area=15)
    assert comp["assemble_hr"].shape == (3,)
    assert np.all((comp["C"] >= 0) & (comp["C"] <= 1))


def test_complementarity_rejects_misaligned_areas():
    net = build_network(_road_trail(), connect=50)
    patches = gpd.GeoDataFrame(geometry=[Point(250, 30)], crs=CRS)
    res = accessibility(net, patches, origins_xy=[(0, 0)])
    with pytest.raises(ValueError):
        complementarity(res["net"], areas=[1, 2, 3])
