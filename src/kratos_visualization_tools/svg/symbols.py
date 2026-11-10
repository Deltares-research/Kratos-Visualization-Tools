"""Support, load, phreatic-line, ruler and axis symbols."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Callable, List, Tuple

from .document import SVG_NS
from .geometry import _line_len, _sample_polyline_points


def _collect_vertical_side_samples(
    polylines_svg: List[List[Tuple[float, float]]],
    side_sign: float,
) -> List[Tuple[float, float, float, float]]:
    candidates: List[Tuple[float, float, float, float]] = []
    for poly in polylines_svg:
        if len(poly) < 2:
            continue
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len = _line_len(dx, dy)
            if seg_len < 1e-9:
                continue
            tx = dx / seg_len
            ty = dy / seg_len
            # Keep only mostly vertical segments from side boundaries.
            if abs(tx) > 0.35:
                continue
            mx = 0.5 * (x1 + x2)
            my = 0.5 * (y1 + y2)
            candidates.append((mx, my, tx, ty))
    # Ensure deterministic top-to-bottom placement.
    candidates.sort(key=lambda t: t[1])
    return candidates


def _pick_two_samples(candidates: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates[0]]
    i1 = max(0, int(round((len(candidates) - 1) * 0.25)))
    i2 = max(0, int(round((len(candidates) - 1) * 0.75)))
    if i1 == i2:
        i2 = min(len(candidates) - 1, i1 + 1)
    return [candidates[i1], candidates[i2]]


def _draw_bottom_hatching(
    group: ET.Element,
    polylines_svg: List[List[Tuple[float, float]]],
    center_svg: Tuple[float, float],
    spacing: float = 22.0,
    hatch_len: float = 12.0,
) -> None:
    cx, cy = center_svg
    for poly in polylines_svg:
        for px, py, tx, ty in _sample_polyline_points(poly, spacing):
            n1x, n1y = ty, -tx
            n2x, n2y = -ty, tx
            to_center_x = cx - px
            to_center_y = cy - py
            d1 = n1x * to_center_x + n1y * to_center_y
            d2 = n2x * to_center_x + n2y * to_center_y
            # Outward normal points away from the domain center.
            nx, ny = (n1x, n1y) if d1 < d2 else (n2x, n2y)
            angle = math.radians(30.0)
            rx = nx * math.cos(angle) - ny * math.sin(angle)
            ry = nx * math.sin(angle) + ny * math.cos(angle)
            ex = px + rx * hatch_len
            ey = py + ry * hatch_len
            ET.SubElement(group, f"{{{SVG_NS}}}line", {
                "x1": f"{px:.3f}",
                "y1": f"{py:.3f}",
                "x2": f"{ex:.3f}",
                "y2": f"{ey:.3f}",
                "stroke": "#222",
                "stroke-width": "1.3",
            })


def _draw_side_fixity(
    group: ET.Element,
    polylines_svg: List[List[Tuple[float, float]]],
    outward_sign: float,
) -> None:
    tri_side = 24.0
    tri_half = 0.5 * tri_side
    tri_depth = tri_side * math.sqrt(3.0) / 2.0
    circle_gap = 12.0
    wheel_r = 3.2
    candidates = _collect_vertical_side_samples(polylines_svg, outward_sign)
    for px, py, tx, ty in _pick_two_samples(candidates):
        nx, ny = outward_sign, 0.0
        bcx = px + nx * tri_depth
        bcy = py + ny * tri_depth
        # Equilateral triangle with vertical side and apex touching boundary.
        b1x = bcx
        b1y = bcy - tri_half
        b2x = bcx
        b2y = bcy + tri_half

        ET.SubElement(group, f"{{{SVG_NS}}}polygon", {
            "points": f"{px:.3f},{py:.3f} {b1x:.3f},{b1y:.3f} {b2x:.3f},{b2y:.3f}",
            "fill": "none",
            "stroke": "#222",
            "stroke-width": "1.2",
        })

        # Two side circles are stacked vertically and touch the triangle side.
        cx = bcx + nx * wheel_r
        cy = bcy
        s1x = cx
        s1y = cy - 0.5 * circle_gap
        s2x = cx
        s2y = cy + 0.5 * circle_gap
        ET.SubElement(group, f"{{{SVG_NS}}}circle", {
            "cx": f"{s1x:.3f}", "cy": f"{s1y:.3f}", "r": f"{wheel_r}",
            "fill": "none", "stroke": "#222", "stroke-width": "1.0",
        })
        ET.SubElement(group, f"{{{SVG_NS}}}circle", {
            "cx": f"{s2x:.3f}", "cy": f"{s2y:.3f}", "r": f"{wheel_r}",
            "fill": "none", "stroke": "#222", "stroke-width": "1.0",
        })

        # Vertical tangent line touching both circles on the outer side.
        vx = cx + nx * wheel_r
        vy1 = cy - tri_half
        vy2 = cy + tri_half
        ET.SubElement(group, f"{{{SVG_NS}}}line", {
            "x1": f"{vx:.3f}",
            "y1": f"{vy1:.3f}",
            "x2": f"{vx:.3f}",
            "y2": f"{vy2:.3f}",
            "stroke": "#222",
            "stroke-width": "1.0",
        })

        # Three short hatch lines attached to the vertical line.
        hatch_len = 6.0
        hatch_drop = 3.0
        for yy in (vy1 + (vy2 - vy1) * 0.1, vy1 + (vy2 - vy1) * 0.5, vy1 + (vy2 - vy1) * 0.9):
            ET.SubElement(group, f"{{{SVG_NS}}}line", {
                "x1": f"{vx:.3f}",
                "y1": f"{yy:.3f}",
                "x2": f"{(vx + nx * hatch_len):.3f}",
                "y2": f"{(yy + hatch_drop):.3f}",
                "stroke": "#222",
                "stroke-width": "1.0",
            })


def _draw_load_arrows(
    group: ET.Element,
    polylines_svg: List[List[Tuple[float, float]]],
    spacing: float = 55.0,
    arrow_len: float = 26.0,
) -> None:
    large_space_threshold = 2.0 * spacing
    ah = 6.0
    aw = 4.0
    
    # Process each load curve separately
    for poly in polylines_svg:
        if len(poly) < 2:
            continue
            
        arrow_positions: List[Tuple[float, float]] = []
        
        # Add arrow at the first endpoint
        arrow_positions.append(poly[0])
        
        # Process segments and add intermediate arrows if needed
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            
            dx = x2 - x1
            dy = y2 - y1
            dist = _line_len(dx, dy)
            
            # If the gap is large, add intermediate arrows
            if dist > large_space_threshold:
                num_intervals = int(dist / spacing)
                for j in range(1, num_intervals):
                    t = j / num_intervals
                    ix = x1 + dx * t
                    iy = y1 + dy * t
                    arrow_positions.append((ix, iy))
        
        # Add arrow at the last endpoint
        arrow_positions.append(poly[-1])
        
        # Draw arrows at all collected positions for this polyline
        for px, py in arrow_positions:
            x1, y1 = px, py - arrow_len
            x2, y2 = px, py
            # Draw vertical arrow line
            ET.SubElement(group, f"{{{SVG_NS}}}line", {
                "x1": f"{x1:.3f}", "y1": f"{y1:.3f}",
                "x2": f"{x2:.3f}", "y2": f"{y2:.3f}",
                "stroke": "#1f1f1f", "stroke-width": "1.6",
            })
            # Draw arrowhead
            ET.SubElement(group, f"{{{SVG_NS}}}polygon", {
                "points": f"{x2:.3f},{y2:.3f} {x2-aw:.3f},{y2-ah:.3f} {x2+aw:.3f},{y2-ah:.3f}",
                "fill": "#1f1f1f",
            })
        
        # Draw horizontal line on top of arrows for this polyline
        if arrow_positions:
            arrow_tops_y = [py - arrow_len for px, py in arrow_positions]
            min_y = min(arrow_tops_y)
            arrow_x_coords = [px for px, py in arrow_positions]
            min_x = min(arrow_x_coords)
            max_x = max(arrow_x_coords)
            ET.SubElement(group, f"{{{SVG_NS}}}line", {
                "x1": f"{min_x:.3f}", "y1": f"{min_y:.3f}",
                "x2": f"{max_x:.3f}", "y2": f"{min_y:.3f}",
                "stroke": "#1f1f1f", "stroke-width": "1.6",
            })


def _draw_phreatic_line(
    group: ET.Element,
    points_svg: List[Tuple[float, float]],
) -> None:
    """Draw a phreatic line as a piece-wise blue dashed line."""
    if len(points_svg) < 2:
        return
    
    path_d = "M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in points_svg)
    ET.SubElement(group, f"{{{SVG_NS}}}path", {
        "d": path_d,
        "fill": "none",
        "stroke": "#0066cc",
        "stroke-width": "2.0",
        "stroke-dasharray": "5,3",
        "stroke-linecap": "round",
    })


def _nice_tick_step(span: float, target_count: int) -> float:
    """Return a 1/2/5 x 10^n tick step giving about ``target_count`` ticks."""
    raw_step = max(span, 1e-12) / max(target_count, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    fraction = raw_step / magnitude
    multiplier = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return multiplier * magnitude


def _format_tick_value(value: float, step: float) -> str:
    """Format a tick value with just enough decimals to resolve ``step``."""
    precision = 0
    while precision < 12 and abs(round(step, precision) - step) > abs(step) * 1e-10:
        precision += 1
    rounded = round(value, precision)
    return f"{rounded + 0.0:.{precision}f}"


def _draw_rulers_and_triad(
    svg: ET.Element,
    width: int,
    height: int,
    padding: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    tr: Callable[[Tuple[float, float]], Tuple[float, float]],
) -> None:
    """Draw model-coordinate rulers along the left/bottom edges and an X/Y triad."""
    left, right = float(padding), float(width - padding)
    top, bottom = float(padding), float(height - padding)
    ruler_color = "#555"
    text_style = {"font-size": "24", "font-family": "DejaVu Sans", "fill": "#333"}
    rulers = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "coordinate_rulers"})
    ET.SubElement(rulers, f"{{{SVG_NS}}}path", {
        "d": f"M{left:.1f},{top:.1f}V{bottom:.1f}H{right:.1f}",
        "fill": "none", "stroke": ruler_color, "stroke-width": "2",
    })

    def ticks(lo: float, hi: float, pixels: float) -> List[Tuple[float, str]]:
        step = _nice_tick_step(hi - lo, max(2, int(pixels / 100)))
        first = math.ceil(lo / step - 1e-10)
        last = math.floor(hi / step + 1e-10)
        return [(i * step, _format_tick_value(i * step, step)) for i in range(first, last + 1)]

    for value, text in ticks(xmin, xmax, right - left):
        x = tr((value, ymin))[0]
        ET.SubElement(rulers, f"{{{SVG_NS}}}line", {
            "x1": f"{x:.2f}", "y1": f"{bottom:.2f}", "x2": f"{x:.2f}", "y2": f"{bottom - 8:.2f}",
            "stroke": ruler_color, "stroke-width": "1",
        })
        ET.SubElement(rulers, f"{{{SVG_NS}}}text", {
            "x": f"{x:.2f}", "y": f"{bottom + 24:.2f}", "text-anchor": "middle", **text_style,
        }).text = text

    for value, text in ticks(ymin, ymax, bottom - top):
        y = tr((xmin, value))[1]
        ET.SubElement(rulers, f"{{{SVG_NS}}}line", {
            "x1": f"{left:.2f}", "y1": f"{y:.2f}", "x2": f"{left + 8:.2f}", "y2": f"{y:.2f}",
            "stroke": ruler_color, "stroke-width": "1",
        })
        ET.SubElement(rulers, f"{{{SVG_NS}}}text", {
            "x": f"{left - 8:.2f}", "y": f"{y + 8:.2f}", "text-anchor": "end", **text_style,
        }).text = text

    triad = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "axes_triad"})
    ox, oy, length = 24.0, float(height) - 24.0, 36.0
    for axis, ex, ey, color, head, label_xy in (
        ("X", ox + length, oy, "#c62828", ((-8, -4), (-8, 4)), (4, 5)),
        ("Y", ox, oy - length, "#2e7d32", ((-4, 8), (4, 8)), (0, -5)),
    ):
        ET.SubElement(triad, f"{{{SVG_NS}}}line", {
            "x1": f"{ox:.1f}", "y1": f"{oy:.1f}", "x2": f"{ex:.1f}", "y2": f"{ey:.1f}",
            "stroke": color, "stroke-width": "2.5",
        })
        points = [(ex, ey)] + [(ex + dx, ey + dy) for dx, dy in head]
        ET.SubElement(triad, f"{{{SVG_NS}}}polygon", {
            "points": " ".join(f"{x:.1f},{y:.1f}" for x, y in points), "fill": color,
        })
        ET.SubElement(triad, f"{{{SVG_NS}}}text", {
            "x": f"{ex + label_xy[0]:.1f}", "y": f"{ey + label_xy[1]:.1f}",
            "font-size": "13", "font-family": "DejaVu Sans", "font-weight": "bold", "fill": color,
        }).text = axis
