"""GiD contour and domain-boundary drawing primitives."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.tri as mtri


def draw_scalar_isolines(
    ax,
    tri: Optional[mtri.Triangulation],
    scalar: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
    color: str = "black",
    count: int = 12,
    show_labels: bool = False,
):
    """Draw explicitly solid isolines over a scalar triangulation."""
    if tri is None or count < 2:
        return None
    valid = np.isfinite(scalar)
    if np.count_nonzero(valid) < 3:
        return None

    triangles = np.asarray(tri.triangles, dtype=int)
    finite_triangles = np.all(valid[triangles], axis=1)
    if not np.any(finite_triangles):
        return None
    finite_tri = mtri.Triangulation(tri.x, tri.y, triangles[finite_triangles])

    lower = float(np.nanmin(scalar[valid])) if vmin is None else float(vmin)
    upper = float(np.nanmax(scalar[valid])) if vmax is None else float(vmax)
    if lower == upper:
        return None
    levels = np.linspace(lower, upper, count)
    contour_scalar = np.asarray(scalar, dtype=float).copy()
    contour_scalar[~valid] = 0.0
    contours = ax.tricontour(
        finite_tri,
        contour_scalar,
        levels=levels,
        colors=color,
        linewidths=0.8,
        linestyles="solid",
    )
    if show_labels:
        ax.clabel(contours, inline=True, inline_spacing=3, fontsize=7, fmt="%g")
    return contours


def draw_triangulation_boundary(
    ax,
    tri: Optional[mtri.Triangulation],
    color: str = "dimgray",
    linewidth: float = 1.2,
    linestyle: str = "-",
):
    """Draw edges belonging to only one triangle, i.e. the domain boundary."""
    if tri is None or tri.triangles.size == 0:
        return []

    edge_counts = defaultdict(int)
    for triangle in np.asarray(tri.triangles, dtype=int):
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge_counts[tuple(sorted((int(first), int(second))))] += 1

    artists = []
    for first, second in (edge for edge, count in edge_counts.items() if count == 1):
        artists.extend(
            ax.plot(
                [tri.x[first], tri.x[second]],
                [tri.y[first], tri.y[second]],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        )
    return artists


