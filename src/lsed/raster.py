"""Raster I/O and point sampling -- the glue between GeoTIFFs / vector sources
and the array algorithms in :mod:`lsed.cost`.

This is the ``terra`` boundary of the R package made explicit: read a surface to
a NumPy array (with its affine ``transform``), coarsen it, burn vector sources
into a mask, and sample a surface at point coordinates (R's ``terra::extract`` at
patch centroids). ``rasterio`` is imported lazily so the array algorithms stay
usable without it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "read_grid",
    "write_grid",
    "cellsize",
    "sample",
    "rasterize_mask",
    "aggregate",
    "raster_terms",
]


def read_grid(path, band=1):
    """Read one band of a raster to ``(array, transform, crs)``."""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(band, masked=True)
        arr = np.where(np.ma.getmaskarray(arr), np.nan, arr.data).astype(float)
        return arr, src.transform, src.crs


def write_grid(path, array, transform, crs, nodata=np.nan):
    """Write a single-band float raster."""
    import rasterio

    array = np.asarray(array, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def cellsize(transform):
    """Pixel size (map units) from an affine ``transform`` (assumes square cells)."""
    return float(abs(transform.a))


def sample(array, transform, xy):
    """Sample a grid at point coordinates (nearest cell), R's ``terra::extract``.

    Parameters
    ----------
    array : numpy.ndarray
        The surface.
    transform : affine.Affine
        Its georeferencing transform.
    xy : array_like, shape (n, 2)
        ``(x, y)`` map coordinates.

    Returns
    -------
    numpy.ndarray
        Cell value at each point; ``NaN`` where the point falls off the grid.
    """
    from rasterio.transform import rowcol

    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    rows, cols = rowcol(transform, xy[:, 0], xy[:, 1])
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    nr, nc = array.shape
    out = np.full(len(xy), np.nan)
    inside = (rows >= 0) & (rows < nr) & (cols >= 0) & (cols < nc)
    out[inside] = array[rows[inside], cols[inside]]
    return out


def rasterize_mask(geometries, out_shape, transform, all_touched=True):
    """Burn vector geometries into a boolean mask aligned to a grid.

    Mirrors ``terra::rasterize`` for source burning: True on every cell the
    geometries cover (``all_touched=True`` catches thin lines / points).

    Parameters
    ----------
    geometries : iterable
        Shapely geometries or a GeoSeries (same CRS as ``transform``).
    out_shape : tuple of int
        ``(rows, cols)``.
    transform : affine.Affine
        Target grid transform.
    all_touched : bool, optional
        Include every cell a geometry touches (default True).

    Returns
    -------
    numpy.ndarray of bool
        The source mask.
    """
    from rasterio.features import rasterize

    geoms = list(getattr(geometries, "geometry", geometries))
    burned = rasterize(
        ((g, 1) for g in geoms),
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    return burned.astype(bool)


def aggregate(array, transform, factor, func="mean"):
    """Coarsen a grid by an integer ``factor`` (R's ``terra::aggregate``).

    Partial edge blocks use whatever cells are present (NaN-aware), and the
    returned transform has its pixel size scaled by ``factor`` with the origin
    unchanged.

    Returns
    -------
    tuple
        ``(coarsened_array, new_transform)``.
    """
    from affine import Affine

    if factor <= 1:
        return np.asarray(array, dtype=float), transform
    a = np.asarray(array, dtype=float)
    nr, nc = a.shape
    pr = (-nr) % factor
    pc = (-nc) % factor
    if pr or pc:
        a = np.pad(a, ((0, pr), (0, pc)), constant_values=np.nan)
    h, w = a.shape
    blocks = a.reshape(h // factor, factor, w // factor, factor)
    reducer = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax, "min": np.nanmin}[func]
    with np.errstate(invalid="ignore"):
        out = reducer(blocks, axis=(1, 3))
    return out, transform @ Affine.scale(factor)


def raster_terms(
    patches,
    suitability,
    transform,
    occ,
    *,
    resistance=None,
    w_habitat=5.0,
    agg=10,
):
    """Compute the raster ranking terms (Dg, Dr) and sample them at centroids.

    The glue that mirrors ``rank_calculate``'s raster block: coarsen the
    suitability surface, build the geographic- and resistance-distance surfaces
    (:mod:`lsed.cost`), and sample each at the patch centroids -- returning a
    ``{"Dg": ..., "Dr": ...}`` dict ready to pass to :func:`lsed.rank.rank_calculate`
    as ``extra_terms``.

    Parameters
    ----------
    patches : geopandas.GeoDataFrame
        Units to score (CRS must match ``transform``).
    suitability : numpy.ndarray
        Habitat-suitability surface on the grid of ``transform``.
    transform : affine.Affine
        Georeferencing of ``suitability``.
    occ : iterable or geopandas.GeoDataFrame
        Known-population source geometries for Dg / Dr.
    resistance : numpy.ndarray, optional
        Precomputed friction; defaults to inverted suitability via
        :func:`lsed.cost.build_resistance`.
    w_habitat : float, optional
        Habitat weight when building the default resistance.
    agg : int, optional
        Coarsen factor for the surfaces (default 10).

    Returns
    -------
    dict
        ``{"Dg": array, "Dr": array}`` indexed to ``patches`` order (inverted so
        near = high, matching the R term orientation).
    """
    from .cost import build_resistance, cost_distance, geo_distance

    suit = np.asarray(suitability, dtype=float)
    suit, tr = aggregate(suit, transform, agg) if agg > 1 else (suit, transform)
    cs = cellsize(tr)

    cen = patches.geometry.centroid
    xy = np.column_stack([cen.x.to_numpy(), cen.y.to_numpy()])

    src = rasterize_mask(occ, suit.shape, tr)

    dg = geo_distance(src, cs, invert=True)

    if resistance is None:
        fric = build_resistance(suit.shape, habitat=(1.0 - suit), w_habitat=w_habitat)
    else:
        fric = np.asarray(resistance, dtype=float)
    dr = cost_distance(fric, src, invert=True)

    return {"Dg": sample(dg, tr, xy), "Dr": sample(dr, tr, xy)}
