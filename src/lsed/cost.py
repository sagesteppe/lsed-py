"""Distance and movement-cost surfaces for the Dg / Dr / A terms.

A Python mirror of the raster engine in ``fn_cost.R``. The functions operate on
NumPy arrays (a grid plus its ``cellsize`` in map units), which is where the
algorithms live; GeoTIFF read/write (R's ``terra`` I/O) is a thin boundary left
to ``rasterio`` and not required here.

terra -> Python mapping:
  terra::distance  (Euclidean)  -> scipy.ndimage.distance_transform_edt
  terra::costDist  (accumulated cost) -> skimage.graph.MCP_Geometric
  terra::terrain   (slope)      -> numpy.gradient
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = [
    "rescale01_grid",
    "tobler_friction",
    "build_resistance",
    "geo_distance",
    "cost_distance",
]


def rescale01_grid(r, invert=False):
    """Min-max a grid to ``[0, 1]`` over its finite cells (NaN-preserving).

    A constant or all-NaN grid maps to zeros. With ``invert=True`` the result is
    ``1 - rescaled`` (near / low-cost becomes high), the index form.
    """
    r = np.asarray(r, dtype=float)
    fin = np.isfinite(r)
    if not fin.any():
        return r * 0.0
    mn = r[fin].min()
    mx = r[fin].max()
    out = (r - mn) / (mx - mn) if mx > mn else r * 0.0
    return 1.0 - out if invert else out


def tobler_friction(dem, cellsize):
    """Isotropic Tobler walking-cost friction from a DEM.

    Walking speed ``W = 6 * exp(-3.5 * |S + 0.05|)`` km/h on slope ``S``
    (rise/run), so cost per unit distance is ``1 / W``. Symmetric in ``|S|`` --
    appropriate for access (a round trip).

    Parameters
    ----------
    dem : numpy.ndarray
        Elevation grid (projected, metres).
    cellsize : float
        Cell size in metres.

    Returns
    -------
    numpy.ndarray
        Relative travel cost (higher = slower going).
    """
    dem = np.asarray(dem, dtype=float)
    gy, gx = np.gradient(dem, cellsize)
    slope = np.hypot(gx, gy)  # rise/run == tan(slope angle)
    speed = 6.0 * np.exp(-3.5 * np.abs(slope + 0.05))
    return 1.0 / speed


def build_resistance(
    shape,
    oceans=None,
    lakes=None,
    rivers=None,
    tri=None,
    habitat=None,
    w_ocean=100.0,
    w_lakes=50.0,
    w_rivers=20.0,
    w_tri=1.0,
    w_habitat=1.0,
    extra=None,
):
    """Isotropic resistance (friction) surface as a weighted sum of layers.

    Sums weighted water masks, z-scored terrain ruggedness, habitat and any
    extras into a per-cell movement cost, clamped to a floor of 1 and returned as
    integer. Mirrors ``buildResistanceSurface`` (safeHavens / ``fn_cost.R``).

    Parameters
    ----------
    shape : tuple of int
        Output grid shape ``(rows, cols)``.
    oceans, lakes, rivers : numpy.ndarray, optional
        Binary (0/1) water-feature masks.
    tri : numpy.ndarray, optional
        Continuous terrain-ruggedness grid (z-scored internally, NaN-robust).
    habitat : numpy.ndarray, optional
        Continuous grid whose *high* values raise cost; pass inverted suitability
        (``1 - suitability``) so movement through good habitat is cheap.
    w_ocean, w_lakes, w_rivers, w_tri, w_habitat : float
        Per-layer weights.
    extra : list of tuple, optional
        ``[(grid, weight), ...]`` additional layers.

    Returns
    -------
    numpy.ndarray
        Integer movement cost, minimum 1.
    """
    res = np.zeros(shape, dtype=float)
    for mask, w in ((oceans, w_ocean), (lakes, w_lakes), (rivers, w_rivers)):
        if mask is not None:
            res = res + w * (np.asarray(mask) == 1)
    if tri is not None:
        tv = np.asarray(tri, dtype=float)
        z = (tv - np.nanmean(tv)) / np.nanstd(tv)
        res = res + w_tri * z
    if habitat is not None:
        res = res + w_habitat * np.asarray(habitat, dtype=float)
    if extra:
        for grid, w in extra:
            res = res + w * np.asarray(grid, dtype=float)
    if np.isfinite(res).any():
        res = np.clip(res, 1, None)
    return np.rint(res).astype(int)


def geo_distance(source_mask, cellsize, invert=False):
    """Euclidean distance to the nearest source cell, rescaled to ``[0, 1]``.

    Parameters
    ----------
    source_mask : numpy.ndarray of bool
        True on source cells.
    cellsize : float
        Cell size in map units (metres); scales the distance.
    invert : bool, optional
        If True, return ``1 - rescaled`` (near = high), the index form.

    Returns
    -------
    numpy.ndarray
        Rescaled distance grid on ``[0, 1]``.
    """
    source_mask = np.asarray(source_mask, dtype=bool)
    d = distance_transform_edt(~source_mask) * cellsize
    return rescale01_grid(d, invert)


def cost_distance(friction, source_mask, invert=False):
    """Accumulated least-cost distance from sources over a friction surface.

    Burns the sources in as zero-cost seeds and accumulates isotropic movement
    cost outward (``skimage.graph.MCP_Geometric``, matching ``terra::costDist``:
    a step costs the geometric distance times the mean of the two cells' costs).
    The result is rescaled to ``[0, 1]``.

    Parameters
    ----------
    friction : numpy.ndarray
        Positive movement-cost grid (e.g. :func:`build_resistance` or
        :func:`tobler_friction`).
    source_mask : numpy.ndarray of bool
        True on source cells (occupied populations for Dr, the network for A).
    invert : bool, optional
        If True, return ``1 - rescaled`` (near / accessible = high).

    Returns
    -------
    numpy.ndarray
        Rescaled accumulated-cost grid on ``[0, 1]``.
    """
    from skimage.graph import MCP_Geometric  # lazy: heavy optional import

    friction = np.asarray(friction, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    if not source_mask.any():
        raise ValueError("source_mask has no source cells")
    fr = friction.copy()
    fr[source_mask] = 0.0  # sources are zero-cost targets (matches terra)
    starts = list(zip(*np.where(source_mask)))
    cum, _ = MCP_Geometric(fr).find_costs(starts)
    cum = np.asarray(cum, dtype=float)
    cum[~np.isfinite(cum)] = np.nan  # unreachable -> NaN
    return rescale01_grid(cum, invert)
