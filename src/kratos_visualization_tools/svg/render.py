"""SVG rendering of GEO/MDPA models and MSH meshes."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import GeoModel, MshModel, Point, Segment
from .document import SVG_NS, _pick_pattern, _pattern_defs, _new_svg
from .msh_reader import _mesh_corner_count, _line_corner_count
from .geometry import _entity_polyline, _loop_to_polygon_points, _curve_points_by_name, _point_key, _segment_key, _chain_segments, _path_data, _polygon_edges, _svg_transform
from .symbols import _draw_bottom_hatching, _draw_side_fixity, _draw_load_arrows, _draw_phreatic_line, _draw_rulers_and_triad
from .legend import _unique_group_ids, _surface_regions, _build_legend_spec


def geo_to_svg(
    model: GeoModel,
    out_path: Path,
    width: int,
    height: int,
    padding: int,
    legend_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    if not model.points:
        raise ValueError("No points parsed from GEO file.")

    xs = [p[0] for p in model.points.values()]
    ys = [p[1] for p in model.points.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    tr = _svg_transform(xmin, xmax, ymin, ymax, width, height, padding)
    svg = _new_svg(width, height)

    _pattern_defs(svg)

    g_surfaces = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "physical_surfaces"})
    g_curves = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "physical_curves"})
    g_symbols = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "supports_and_loads"})
    g_legend = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "legend"})

    surface_names = sorted(model.physical_surfaces.keys())
    curve_names = sorted(model.physical_curves.keys())
    legend_spec = _build_legend_spec(model, legend_cfg)
    center_svg = tr((0.5 * (xmin + xmax), 0.5 * (ymin + ymax)))
    surface_ids = _unique_group_ids(model.physical_surfaces, legend_spec.surface_entries)
    curve_ids = _unique_group_ids(model.physical_curves, legend_spec.curve_entries)

    # Each region is one Physical Surface, or all names grouped in one legend
    # entry. A region's fill is a single path (no seams between polygons) and
    # only its outer boundary is stroked; shared internal edges are suppressed.
    regions = _surface_regions(surface_names, legend_spec.surface_style_by_name)
    for style_key, region_names in regions:
        color = legend_spec.color(style_key)
        pat = _pick_pattern(style_key)

        all_polys: List[List[Tuple[float, float]]] = []
        for name in region_names:
            for sid in surface_ids[name]:
                loop_ids = model.plane_surfaces.get(abs(sid), [])
                for loop_id in loop_ids:
                    poly = _loop_to_polygon_points(model, abs(loop_id))
                    if len(poly) < 4:
                        continue
                    all_polys.append(poly)

        if not all_polys:
            continue

        # Outer boundary: edges that appear in exactly one polygon of this
        # region. Internal shared edges are counted twice and suppressed.
        edge_count: Dict[Tuple[Point, Point], int] = {}
        edge_world: Dict[Tuple[Point, Point], Segment] = {}
        for poly in all_polys:
            for edge in _polygon_edges(poly):
                key = _segment_key(edge)
                edge_count[key] = edge_count.get(key, 0) + 1
                edge_world.setdefault(key, edge)
        boundary = [edge_world[key] for key, cnt in edge_count.items() if cnt == 1]
        chains = _chain_segments(boundary)
        outlines = [[tr(p) for p in chain] for chain in chains]

        # With evenodd, the closed boundary loops fill exactly the polygons'
        # union (minus holes), so the per-element polygons are not written.
        # Fall back to them if the boundary does not form closed loops.
        if chains and all(_point_key(c[0]) == _point_key(c[-1]) for c in chains):
            fill_d = _path_data(outlines)
        else:
            fill_d = _path_data([[tr(p) for p in poly] for poly in all_polys])
        for fill, opacity in ((color, "0.25"), (f"url(#{pat})", "0.45")):
            ET.SubElement(g_surfaces, f"{{{SVG_NS}}}path", {
                "d": fill_d,
                "fill": fill,
                "fill-opacity": opacity,
                "fill-rule": "evenodd",
                "stroke": "none",
            })

        if outlines:
            ET.SubElement(g_surfaces, f"{{{SVG_NS}}}path", {
                "d": _path_data(outlines),
                "fill": "none",
                "stroke": color,
                "stroke-width": "1.2",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            })

    # One path per curve group; coincident segments (e.g. both faces of a
    # zero-thickness interface) are drawn once and connected ones are chained.
    for name in curve_names:
        segments: Dict[Tuple[Point, Point], Segment] = {}
        for cid in curve_ids[name]:
            seg = _entity_polyline(model, cid)
            for p1, p2 in zip(seg, seg[1:]):
                key = _segment_key((p1, p2))
                if key[0] != key[1]:
                    segments.setdefault(key, (p1, p2))
        path_d = _path_data([[tr(p) for p in chain] for chain in _chain_segments(list(segments.values()))])
        if not path_d:
            continue
        style_key = legend_spec.curve_style_by_name.get(name, name)
        ET.SubElement(g_curves, f"{{{SVG_NS}}}path", {
            "d": path_d,
            "fill": "none",
            "stroke": legend_spec.color(style_key),
            "stroke-width": "2.5",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        })

    symbols_cfg = (legend_cfg or {}).get("symbols", {}) if isinstance(legend_cfg, dict) else {}

    bottom_names = symbols_cfg.get("bottom_hatch_curves", ["Bottom"])
    if isinstance(bottom_names, list):
        bottom_polys = _curve_points_by_name(model, [str(n) for n in bottom_names])
        _draw_bottom_hatching(
            g_symbols,
            [[tr(p) for p in poly] for poly in bottom_polys],
            center_svg=center_svg,
        )

    left_names = symbols_cfg.get("left_fixity_curves", ["ground_left_boundary"])
    if isinstance(left_names, list):
        left_polys = _curve_points_by_name(model, [str(n) for n in left_names])
        _draw_side_fixity(g_symbols, [[tr(p) for p in poly] for poly in left_polys], outward_sign=-1.0)

    right_names = symbols_cfg.get("right_fixity_curves", ["ground_right_boundary"])
    if isinstance(right_names, list):
        right_polys = _curve_points_by_name(model, [str(n) for n in right_names])
        _draw_side_fixity(g_symbols, [[tr(p) for p in poly] for poly in right_polys], outward_sign=1.0)

    load_names = symbols_cfg.get("load_arrow_curves", ["kraan_load_p1", "kraan_load_p2", "water_loads"])
    if isinstance(load_names, list):
        load_polys = _curve_points_by_name(model, [str(n) for n in load_names])
        _draw_load_arrows(g_symbols, [[tr(p) for p in poly] for poly in load_polys])

    # Draw phreatic line if configured
    if legend_spec.phreatic_line_points:
        phreatic_points_svg = [tr(p) for p in legend_spec.phreatic_line_points]
        _draw_phreatic_line(g_symbols, phreatic_points_svg)

    if legend_spec.show_rulers:
        _draw_rulers_and_triad(svg, width, height, padding, xmin, xmax, ymin, ymax, tr)

    # Place legend using explicit top-left coordinates from config.
    # Spacing and swatches scale with the label font (11 px reproduces the defaults).
    lx = float(legend_spec.legend_top_left[0])
    ly = float(legend_spec.legend_top_left[1])
    title_fs = f"{legend_spec.title_font_size:g}"
    label_fs = f"{legend_spec.label_font_size:g}"
    k = legend_spec.label_font_size / 11.0
    title_gap = legend_spec.title_font_size + 4.0 * k
    row_h = 16.0 * k
    text_dx = 24.0 * k
    lx2 = lx + 360.0 * k

    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx), "y": str(ly), "font-size": title_fs, "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Physical Curves (conditions)"

    y = ly + title_gap
    curve_legend = legend_spec.curve_entries[: max(0, legend_spec.max_curve_items)]
    for entry in curve_legend:
        color = legend_spec.color(entry.style_key)
        ET.SubElement(g_legend, f"{{{SVG_NS}}}line", {
            "x1": str(lx), "y1": str(y), "x2": str(lx + 18 * k), "y2": str(y),
            "stroke": color, "stroke-width": "3"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx + text_dx), "y": str(y + 4 * k), "font-size": label_fs, "font-family": "DejaVu Sans", "fill": "#111"
        }).text = entry.label
        y += row_h

    # Draw phreatic line in legend if provided
    if legend_spec.phreatic_line_label and legend_spec.phreatic_line_points:
        y += 8 * k  # Add extra space before phreatic line
        ET.SubElement(g_legend, f"{{{SVG_NS}}}line", {
            "x1": str(lx), "y1": str(y - 4 * k), "x2": str(lx + 18 * k), "y2": str(y - 4 * k),
            "stroke": "#0066cc", "stroke-width": "2", "stroke-dasharray": "5,3"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx + text_dx), "y": str(y), "font-size": label_fs, "font-family": "DejaVu Sans", "fill": "#111"
        }).text = legend_spec.phreatic_line_label

    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx2), "y": str(ly), "font-size": title_fs, "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Physical Surfaces (elements)"

    y2 = ly + title_gap
    surface_legend = legend_spec.surface_entries[: max(0, legend_spec.max_surface_items)]
    for entry in surface_legend:
        color = legend_spec.color(entry.style_key)
        pat = _pick_pattern(entry.style_key)
        swatch = {
            "x": str(lx2), "y": str(y2 - 10 * k),
            "width": f"{16 * k:g}", "height": f"{12 * k:g}",
        }
        ET.SubElement(g_legend, f"{{{SVG_NS}}}rect", {
            **swatch, "fill": color, "fill-opacity": "0.25", "stroke": color, "stroke-width": "1"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}rect", {
            **swatch, "fill": f"url(#{pat})", "fill-opacity": "0.45", "stroke": "none"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx2 + text_dx), "y": str(y2), "font-size": label_fs, "font-family": "DejaVu Sans", "fill": "#111"
        }).text = entry.label
        y2 += row_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(out_path, encoding="utf-8", xml_declaration=True)


def msh_to_svg(
    model: MshModel,
    out_path: Path,
    width: int,
    height: int,
    padding: int,
    legend_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    if not model.nodes:
        raise ValueError("No nodes parsed from MSH file.")

    # Styling for .msh mode can be configured from legend JSON under "msh".
    mesh_cfg = legend_cfg.get("msh", {}) if isinstance(legend_cfg, dict) else {}
    if not isinstance(mesh_cfg, dict):
        mesh_cfg = {}

    shared_edges_black = bool(mesh_cfg.get("shared_edges_black", True))
    shared_edge_color = str(mesh_cfg.get("shared_edge_color", "#000000"))
    shared_edge_width = float(mesh_cfg.get("shared_edge_width", 1.0))
    boundary_edge_width = float(mesh_cfg.get("boundary_edge_width", 0.8))

    polygons: List[Tuple[int, List[int], List[Tuple[float, float]], List[int]]] = []
    for eid, etype, node_ids, phys_tags in model.elements:
        ncorner = _mesh_corner_count(etype)
        if ncorner < 3 or len(node_ids) < ncorner:
            continue
        corners = node_ids[:ncorner]
        if any(nid not in model.nodes for nid in corners):
            continue
        pts2d = [(model.nodes[nid][0], model.nodes[nid][1]) for nid in corners]
        polygons.append((eid, corners, pts2d, phys_tags))

    if not polygons:
        raise ValueError("No 2D mesh elements (triangles/quads) parsed from MSH file.")

    xs = [x for _, _, poly, _ in polygons for x, _ in poly]
    ys = [y for _, _, poly, _ in polygons for _, y in poly]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    tr = _svg_transform(xmin, xmax, ymin, ymax, width, height, padding)
    svg = _new_svg(width, height)

    edge_to_elements: Dict[Tuple[int, int], List[int]] = {}
    edge_coords: Dict[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    element_edges: Dict[int, List[Tuple[int, int]]] = {}

    # Build surface/curve style maps from legend-config entries.
    surface_names = sorted(
        name for _ptag, (dim, name) in model.physical_names.items() if dim == 2 and name
    )
    curve_names = sorted(
        name for _ptag, (dim, name) in model.physical_names.items() if dim == 1 and name
    )
    fake_geo = GeoModel(
        physical_curves={name: [] for name in curve_names},
        physical_surfaces={name: [] for name in surface_names},
    )
    legend_spec = _build_legend_spec(fake_geo, legend_cfg)

    def physical_name_from_tags(phys_tags: List[int], dimension: int) -> Optional[str]:
        for ptag in phys_tags:
            info = model.physical_names.get(ptag)
            if info and info[0] == dimension:
                return info[1]
        return None

    elem_color: Dict[int, str] = {}
    elem_style_key: Dict[int, str] = {}
    condition_color_by_edge: Dict[Tuple[int, int], str] = {}

    # Map boundary edges to condition colors from 1D physical groups.
    for _eid, etype, node_ids, phys_tags in model.line_elements:
        ncorner = _line_corner_count(etype)
        if ncorner < 2 or len(node_ids) < 2:
            continue
        n1, n2 = node_ids[0], node_ids[1]
        key = (n1, n2) if n1 < n2 else (n2, n1)
        cname = physical_name_from_tags(phys_tags, 1)
        if not cname:
            continue
        style_key = legend_spec.curve_style_by_name.get(cname, cname)
        ccolor = legend_spec.color(style_key)
        # Keep first assignment for deterministic output when duplicates exist.
        if key not in condition_color_by_edge:
            condition_color_by_edge[key] = ccolor

    for eid, corners, _poly, phys_tags in polygons:
        sname = physical_name_from_tags(phys_tags, 2)
        if sname:
            style_key = legend_spec.surface_style_by_name.get(sname, sname)
        else:
            style_key = f"elem_{eid}"
        elem_style_key[eid] = style_key
        elem_color[eid] = legend_spec.color(style_key)

        local_edges: List[Tuple[int, int]] = []
        for i in range(len(corners)):
            n1 = corners[i]
            n2 = corners[(i + 1) % len(corners)]
            key = (n1, n2) if n1 < n2 else (n2, n1)
            local_edges.append(key)
            edge_to_elements.setdefault(key, []).append(eid)
            if key not in edge_coords:
                p1 = (model.nodes[n1][0], model.nodes[n1][1])
                p2 = (model.nodes[n2][0], model.nodes[n2][1])
                edge_coords[key] = (p1, p2)
        element_edges[eid] = local_edges

    shared_edges = {e for e, owners in edge_to_elements.items() if len(owners) > 1}

    g_elements = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "mesh_elements"})
    g_shared = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "mesh_shared_edges"})
    g_legend = ET.SubElement(svg, f"{{{SVG_NS}}}g", {"id": "legend"})

    # Draw boundary edges per element color (surface style if available).
    # Group edges by (color, stroke_width) to emit one <path> per group instead of
    # one <line> per edge — dramatically reduces SVG element count.
    boundary_paths: Dict[Tuple[str, float], List[str]] = {}
    for eid, _corners, _poly, _phys_tags in polygons:
        default_color = elem_color[eid]
        for edge in element_edges.get(eid, []):
            if edge in shared_edges:
                continue
            color = condition_color_by_edge.get(edge, default_color)
            p1, p2 = edge_coords[edge]
            x1, y1 = tr(p1)
            x2, y2 = tr(p2)
            boundary_paths.setdefault((color, boundary_edge_width), []).append(
                f"M{x1:.1f},{y1:.1f}L{x2:.1f},{y2:.1f}"
            )
    for (color, stroke_width), segs in boundary_paths.items():
        ET.SubElement(g_elements, f"{{{SVG_NS}}}path", {
            "d": " ".join(segs),
            "stroke": color,
            "stroke-width": f"{stroke_width:.2f}",
            "fill": "none",
            "stroke-linecap": "round",
        })

    # Draw shared edges: keep color for same-surface neighbors, black on interfaces.
    # Group by (color, stroke_width) to emit one <path> per group.
    shared_paths: Dict[Tuple[str, float], List[str]] = {}
    for edge in sorted(shared_edges):
        owners = edge_to_elements.get(edge, [])
        owner_styles = {elem_style_key[eid] for eid in owners if eid in elem_style_key}
        owner_colors = {elem_color[eid] for eid in owners if eid in elem_color}

        if owner_colors and (len(owner_styles) <= 1 or not shared_edges_black):
            stroke = next(iter(owner_colors))
            width_val = boundary_edge_width
        else:
            stroke = shared_edge_color
            width_val = shared_edge_width

        p1, p2 = edge_coords[edge]
        x1, y1 = tr(p1)
        x2, y2 = tr(p2)
        shared_paths.setdefault((stroke, width_val), []).append(
            f"M{x1:.1f},{y1:.1f}L{x2:.1f},{y2:.1f}"
        )
    for (stroke, stroke_width), segs in shared_paths.items():
        ET.SubElement(g_shared, f"{{{SVG_NS}}}path", {
            "d": " ".join(segs),
            "stroke": stroke,
            "stroke-width": f"{stroke_width:.2f}",
            "fill": "none",
            "stroke-linecap": "round",
        })

    if legend_spec.show_rulers:
        _draw_rulers_and_triad(svg, width, height, padding, xmin, xmax, ymin, ymax, tr)

    lx = 20
    ly = 24
    title_fs = f"{legend_spec.title_font_size:g}"
    label_fs = f"{legend_spec.label_font_size:g}"
    k = legend_spec.label_font_size / 11.0
    lines = [
        f"Elements rendered: {len(polygons)} (line-only view)",
        "Edges: surface style colors",
        "Boundary conditions: curve style colors",
    ]
    if shared_edges_black:
        lines.append("Interfaces between surface styles: black")
    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx), "y": str(ly), "font-size": title_fs, "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Mesh view (.msh input)"
    y = ly + legend_spec.title_font_size + 4.0 * k
    for text in lines:
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx), "y": f"{y:g}", "font-size": label_fs, "font-family": "DejaVu Sans", "fill": "#111"
        }).text = text
        y += 16.0 * k

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(out_path, encoding="utf-8", xml_declaration=True)
