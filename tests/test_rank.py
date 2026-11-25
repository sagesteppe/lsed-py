import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from lsed import rank_calculate, rank_index

CRS = "EPSG:32613"


def test_rank_index_weighted_mean_and_rank():
    out = rank_index({"Dg": [0.0, 1.0, 2.0], "A": [2.0, 1.0, 0.0]})
    # each term rescaled to [0,1]; equal weights -> index is their mean
    assert np.allclose(out["Dg"], [0.0, 0.5, 1.0])
    assert np.allclose(out["A"], [1.0, 0.5, 0.0])
    assert np.allclose(out["index"], 0.5)  # ties everywhere
    assert list(out["rank"]) == [1, 1, 1]  # ties.method="min"


def test_rank_index_is_na_safe_per_row():
    out = rank_index({"Dg": [0.0, 1.0, np.nan], "A": [0.0, 1.0, 1.0]})
    # row 2 scored on A alone (renormalised), not dragged to NaN
    assert not np.isnan(out["index"].iloc[2])
    assert out.loc[2, "rank"] == 1  # highest remaining score


def test_rank_index_all_na_row_is_nan():
    out = rank_index({"Dg": [1.0, np.nan], "A": [0.0, np.nan]})
    assert np.isnan(out["index"].iloc[1])
    assert out["rank"].iloc[1] is pd_na()


def test_rank_index_rejects_unknown_weight():
    with pytest.raises(ValueError):
        rank_index({"Dg": [0.0, 1.0]}, weights={"nope": 2.0})


def test_rank_index_respects_weights():
    out = rank_index({"Dg": [0.0, 1.0], "A": [1.0, 0.0]}, weights={"Dg": 3.0})
    # Dg weighted 3x A: patch1 (Dg=1) should beat patch0
    assert out.loc[1, "index"] > out.loc[0, "index"]
    assert out.loc[1, "rank"] == 1


def _network():
    return gpd.GeoDataFrame(
        {
            "class": ["road", "trail"],
            "highway": ["residential", "path"],
            "geometry": [
                LineString([(0, 0), (200, 0)]),
                LineString([(200, 0), (300, 50)]),
            ],
        },
        crs=CRS,
    )


def test_rank_calculate_end_to_end_with_network_and_p():
    patches = gpd.GeoDataFrame(
        {
            "acres": [10.0, 10.0, 10.0],
            "prob_mean": [0.9, 0.5, 0.2],
            "geometry": [Point(250, 30), Point(260, 35), Point(100, 20)],
        },
        crs=CRS,
    )
    out = rank_calculate(patches, network=_network(), origin_xy=[(0, 0)])
    # A, C from the network + P from prob_mean, all assembled into index/rank
    for col in ("A", "C", "P", "index", "rank"):
        assert col in out.columns
    assert set(out["rank"].dropna()) <= {1, 2, 3}
    assert out["index"].between(0, 1).all()


def test_rank_calculate_accepts_precomputed_raster_terms():
    patches = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]}, crs=CRS
    )
    out = rank_calculate(patches, extra_terms={"Dg": [0.0, 1.0, 2.0]})
    assert "Dg" in out.columns and "index" in out.columns
    assert out.loc[2, "rank"] == 1  # highest Dg -> top priority


def test_rank_calculate_needs_some_term():
    patches = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=CRS)
    with pytest.raises(ValueError):
        rank_calculate(patches, prob_col=None)


def pd_na():
    import pandas as pd

    return pd.NA
