"""Polyline, loop, segment-chaining and coordinate helpers."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

from .models import GeoModel, Point, Segment


def _entity_polyline(model: GeoModel, signed_id: int) -> List[Tuple[float, float]]:
    eid = abs(signed_id)
    nids = model.lines[eid] if eid in model.lines else model.splines.get(eid, [])
    pts = [model.points[n] for n in nids if n in model.points]
    return pts if signed_id > 0 else list(reversed(pts))


def _loop_to_polygon_points(model: GeoModel, loop_id: int) -> List[Tuple[float, float]]:
    seq = model.curve_loops.get(loop_id, [])
    all_pts: List[Tuple[float, float]] = []

    for signed_cid in seq:
        seg = _entity_polyline(model, signed_cid)
        if not seg:
            continue
        if all_pts and seg:
            if all_pts[-1] == seg[0]:
                all_pts.extend(seg[1:])
            elif all_pts[-1] == seg[-1]:
                seg = list(reversed(seg))
                all_pts.extend(seg[1:])
            else:
                all_pts.extend(seg)
        else:
            all_pts.extend(seg)

    if len(all_pts) >= 3 and all_pts[0] != all_pts[-1]:
        all_pts.append(all_pts[0])
    return all_pts


def _curve_points_by_name(model: GeoModel, names: List[str]) -> List[List[Tuple[float, float]]]:
    """Return each group's curves as ordered polylines for symbol placement.

    Segments are deduplicated and chained, so groups stored as unordered
    pieces (e.g. MDPA conditions or node-only edges) give continuous runs
    and symbols are spaced evenly along them.
    """
    out: List[List[Tuple[float, float]]] = []
    for name in names:
        segments: Dict[Tuple[Point, Point], Segment] = {}
        for cid in model.physical_curves.get(name, []):
            pts = _entity_polyline(model, cid)
            for p1, p2 in zip(pts, pts[1:]):
                key = _segment_key((p1, p2))
                if key[0] != key[1]:
                    segments.setdefault(key, (p1, p2))
        out.extend(_chain_segments(list(segments.values())))
    return out


def _line_len(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def _sample_polyline_points(poly: List[Tuple[float, float]], spacing: float) -> List[Tuple[float, float, float, float]]:
    samples: List[Tuple[float, float, float, float]] = []
    if len(poly) < 2:
        return samples

    next_at = spacing
    traveled = 0.0
    for i in range(len(poly) - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = _line_len(dx, dy)
        if seg_len < 1e-12:
            continue

        while traveled + seg_len >= next_at:
            t = (next_at - traveled) / seg_len
            px = x1 + dx * t
            py = y1 + dy * t
            samples.append((px, py, dx / seg_len, dy / seg_len))
            next_at += spacing

        traveled += seg_len

    return samples


def _point_key(point: Point) -> Point:
    return (round(point[0], 9), round(point[1], 9))


def _segment_key(segment: Segment) -> Tuple[Point, Point]:
    a, b = _point_key(segment[0]), _point_key(segment[1])
    return (a, b) if a <= b else (b, a)


def _chain_segments(segments: List[Segment]) -> List[List[Point]]:
    """Join segments that share end points into polylines.

    Closed chains end at their start point. Open chains start at odd-degree
    vertices so that they are not split unnecessarily.
    """
    adjacency: Dict[Point, List[int]] = {}
    point_at: Dict[Point, Point] = {}
    for idx, (a, b) in enumerate(segments):
        for p in (a, b):
            key = _point_key(p)
            adjacency.setdefault(key, []).append(idx)
            point_at.setdefault(key, p)

    used = [False] * len(segments)
    starts = [key for key, ids in adjacency.items() if len(ids) % 2] + list(adjacency)
    chains: List[List[Point]] = []
    for start in starts:
        while True:
            current = start
            chain = [point_at[start]]
            while True:
                ids = adjacency[current]
                while ids and used[ids[-1]]:
                    ids.pop()
                if not ids:
                    break
                idx = ids.pop()
                used[idx] = True
                a, b = segments[idx]
                nxt = b if _point_key(a) == current else a
                chain.append(nxt)
                current = _point_key(nxt)
            if len(chain) < 2:
                break
            chains.append(chain)
    return chains


def _is_on_segment(a: Point, b: Point, c: Point, tolerance: float) -> bool:
    """True if ``b`` lies within ``tolerance`` of the segment from ``a`` to ``c``."""
    dx, dy = c[0] - a[0], c[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return False
    t = ((b[0] - a[0]) * dx + (b[1] - a[1]) * dy) / length_sq
    offset = abs(dx * (b[1] - a[1]) - dy * (b[0] - a[0])) / math.sqrt(length_sq)
    return 0.0 <= t <= 1.0 and offset <= tolerance


def _drop_collinear(points: List[Point], tolerance: float = 0.05) -> List[Point]:
    """Remove points lying within ``tolerance`` (SVG px) of the straight line between neighbours."""
    if len(points) < 3:
        return list(points)
    kept = [points[0]]
    for b, c in zip(points[1:-1], points[2:]):
        if not _is_on_segment(kept[-1], b, c, tolerance):
            kept.append(b)
    kept.append(points[-1])
    closed = math.dist(kept[0], kept[-1]) <= 1e-6
    if closed and len(kept) > 4 and _is_on_segment(kept[-2], kept[0], kept[1], tolerance):
        kept = kept[1:-1] + [kept[1]]
    return kept


def _path_data(polylines: List[List[Point]]) -> str:
    """Compact SVG path data; polylines whose ends coincide are closed with Z."""
    parts: List[str] = []
    for points in polylines:
        coords: List[str] = []
        for x, y in _drop_collinear(points):
            coord = f"{x:.1f},{y:.1f}"
            if not coords or coords[-1] != coord:
                coords.append(coord)
        if len(coords) < 2:
            continue
        closed = len(coords) > 2 and coords[0] == coords[-1]
        if closed:
            coords.pop()
        parts.append("M" + "L".join(coords) + ("Z" if closed else ""))
    return "".join(parts)


def _polygon_edges(poly: List[Point]) -> List[Segment]:
    pts = poly[:-1] if len(poly) > 1 and poly[0] == poly[-1] else list(poly)
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _svg_transform(
    xmin: float, xmax: float, ymin: float, ymax: float,
    width: int, height: int, padding: int,
) -> Callable[[Tuple[float, float]], Tuple[float, float]]:
    span_x = max(xmax - xmin, 1e-9)
    span_y = max(ymax - ymin, 1e-9)

    def tr(pt: Tuple[float, float]) -> Tuple[float, float]:
        x, y = pt
        sx = padding + (x - xmin) / span_x * (width - 2 * padding)
        sy_data = padding + (y - ymin) / span_y * (height - 2 * padding)
        return sx, height - sy_data

    return tr
