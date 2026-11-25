import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from lsed import (
    eval_survey,
    rank_calculate,
    survey_field,
    survey_term,
    survey_weights,
)

CRS = "EPSG:32613"


def test_survey_weights_floor_and_order():
    # species A: 6 records; B: 2 records (below floor of 5)
    coords = np.array([(i, 0) for i in range(6)] + [(0, 0), (1, 0)], dtype=float)
    species = ["A"] * 6 + ["B"] * 2
    w = survey_weights(coords, species)
    assert list(w["species"]) == ["A"]  # B dropped, sorted order
    assert w.loc[0, "n"] == 6
    assert w.loc[0, "phi"] == 1.0  # no domain -> phi 1
    assert np.isclose(w.loc[0, "w"], 1 / 6)  # (phi/n)^1


def test_survey_weights_phi_from_domain_overlap():
    # records spread across x in [0, 10]; domain covers only the left half
    coords = np.array([(x, 0.0) for x in np.linspace(0, 10, 10)])
    # give the box non-zero area with a little y spread
    coords[:, 1] = np.linspace(0, 10, 10)
    species = ["A"] * 10
    dom = box(0, 0, 5, 20)  # covers roughly the lower-x half of the p95 box
    w = survey_weights(coords, species, domain=dom)
    assert 0 < w.loc[0, "phi"] < 1


def test_eval_survey_peaks_near_records_and_truncates():
    field = survey_field(
        np.array([[0.0, 0.0]]), ["A"], sigma=10.0, trunc=4.0, record_type=["Plot"]
    )
    s = eval_survey(field, [(0, 0), (5, 0), (1000, 0)])
    assert s[0] > s[1] > 0  # decays with distance
    assert s[2] == 0.0  # beyond trunc*sigma = 40 m -> truncated to 0


def test_eval_survey_weights_add():
    # two coincident records sum their weights at that point
    field = survey_field(np.array([[0.0, 0.0], [0.0, 0.0]]), ["A", "A"],
                         sigma=10.0, record_type=["Plot", "Plot"])
    # both are plots at the same coord -> deduped to ONE plot point, weight 1
    assert field.coords.shape[0] == 1
    assert np.isclose(eval_survey(field, [(0, 0)])[0], 1.0)  # exp(0)*1


def test_plot_vs_opportunistic_split():
    coords = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 0.0]])
    species = ["A", "A", "A"]
    rtype = ["Plot", "iNat", "iNat"]
    # need >=5 opportunistic A for weights; pad them
    coords = np.vstack([coords, [[200.0 + i, 0.0] for i in range(4)]])
    species = species + ["A"] * 4
    rtype = rtype + ["iNat"] * 4
    field = survey_field(coords, species, record_type=rtype, sigma=50.0)
    assert "__plot__" in set(field.species)
    assert (field.species == "__plot__").sum() == 1  # the one plot point


def test_survey_field_requires_positive_sigma():
    with pytest.raises(ValueError):
        survey_field(np.array([[0.0, 0.0]]), ["A"], sigma=0)


def test_survey_term_feeds_rank_calculate():
    # records cluster near patch 0; patch 1 is unsurveyed
    recs = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    field = survey_field(recs, ["A", "A", "A"], sigma=20.0,
                         record_type=["Plot", "Plot", "Plot"])
    patches = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(500, 500)], crs=CRS)
    s = survey_term(patches, field)
    assert s[0] < s[1]  # surveyed patch gets the more-negative (lower) S
    out = rank_calculate(patches, extra_terms={"S": s})
    assert "S" in out.columns
    # unsurveyed patch 1 ranks first (S rescales: high S = high priority)
    assert out.loc[1, "rank"] == 1
