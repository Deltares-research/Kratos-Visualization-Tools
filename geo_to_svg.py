#!/usr/bin/env python3
"""Convert a Gmsh .geo/.msh or Kratos .mdpa file into an SVG preview.

Features:
- Draws physical curves (conditions) with distinct colors.
- Draws physical surfaces (elements) with hatch/dot patterns.
- If input is .msh, draws each mesh element with a distinct fill color.
- Supports common GEO entities used in 2D profiles: Point, Line, Spline,
  Curve Loop / Line Loop, Plane Surface, Physical Curve/Line/Surface.
- MDPA input reuses the package parser and draws SubModelParts as regions/curves.
- GEO/MSH input requires only the standard library.

Usage:
  python geo_to_svg.py --input model.mdpa --output model.svg
  python Eemdijk/tools/geo_to_svg.py --input Eemdijk/eemdijk.geo --output Eemdijk/Pictures/eemdijk_geo.svg
    python Eemdijk/tools/geo_to_svg.py --input Eemdijk/eemdijk.msh --output Eemdijk/Pictures/eemdijk_mesh.svg

Optional legend control JSON:
    python Eemdijk/tools/geo_to_svg.py --input Eemdijk/eemdijk.geo --output out.svg --legend-config cfg.json

Legend config schema (root object, or nested under "legend"):
    - "curves": [{"name": "group", "label": "Legend name", "show": true}, ...]
    - "surfaces": [{"name": "group"} or {"names": ["g1", "g2"], "label": "Combined"}, ...]
      (grouped names are drawn as one region with one outline)
    - "max_curves": int, "max_surfaces": int
    - "top_left": [x, y] or {"x": x, "y": y} (legend top-left anchor in SVG pixels)
    - "phreatic_line": {"label": "...", "points": [[x1, y1], [x2, y2], ...]}
    - "symbols": {
          "bottom_hatch_curves": ["Bottom"],
          "left_fixity_curves": ["ground_left_boundary"],
          "right_fixity_curves": ["ground_right_boundary"],
          "load_arrow_curves": ["kraan_load_p1", "kraan_load_p2", "water_loads"]
      }
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass
class GeoModel:
    vars: Dict[str, float] = field(default_factory=dict)
    points: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    lines: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    splines: Dict[int, List[int]] = field(default_factory=dict)
    curve_loops: Dict[int, List[int]] = field(default_factory=dict)
    plane_surfaces: Dict[int, List[int]] = field(default_factory=dict)
    physical_curves: Dict[str, List[int]] = field(default_factory=dict)
    physical_surfaces: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class MshModel:
    nodes: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    # Physical tag -> (dimension, name)
    physical_names: Dict[int, Tuple[int, str]] = field(default_factory=dict)
    # Surface entity tag -> physical tags (Gmsh v4).
    surface_entity_phys: Dict[int, List[int]] = field(default_factory=dict)
    # Curve entity tag -> physical tags (Gmsh v4).
    curve_entity_phys: Dict[int, List[int]] = field(default_factory=dict)
    # Stored as (element_id, element_type, node_ids, physical_tags)
    elements: List[Tuple[int, int, List[int], List[int]]] = field(default_factory=list)
    # Stored as (element_id, element_type, node_ids, physical_tags) for 1D mesh entities.
    line_elements: List[Tuple[int, int, List[int], List[int]]] = field(default_factory=list)


def parse_mdpa(path: Path) -> GeoModel:
    """Adapt supported 2D MDPA geometry to named SVG surfaces and curves."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from kratos_visualization_tools.mdpa import _connection_paths, parse_mdpa_data

    parsed = parse_mdpa_data(path)
    if not parsed.nodes:
        raise ValueError("No nodes parsed from MDPA file.")
    if not parsed.entity_records:
        raise ValueError("No elements or conditions parsed from MDPA file.")

    model = GeoModel(points={nid: (x, y) for nid, (x, y, _z) in parsed.nodes.items()})
    for block, entity_type, entity_id, node_ids in parsed.entity_records:
        missing = sorted(set(node_ids) - model.points.keys())
        if missing:
            raise ValueError(f"{block} {entity_id} references missing nodes: {missing}")
        try:
            paths = _connection_paths(entity_type, node_ids)
        except ValueError as error:
            raise ValueError(f"{block} {entity_id}: {error}") from error
        names = sorted(parsed.entity_submodels.get((block, entity_id), ())) or [entity_type]
        for node_path in paths:
            curve_id = len(model.splines) + 1
            model.splines[curve_id] = list(node_path)
            if len(node_path) >= 4 and node_path[0] == node_path[-1]:
                surface_id = len(model.plane_surfaces) + 1
                model.curve_loops[surface_id] = [curve_id]
                model.plane_surfaces[surface_id] = [surface_id]
                for name in names:
                    model.physical_surfaces.setdefault(name, []).append(surface_id)
            else:
                for name in names:
                    model.physical_curves.setdefault(name, []).append(curve_id)
    return model


def _strip_comments(text: str) -> str:
    out = []
    for raw in text.splitlines():
        line = raw
        if "//" in line:
            line = line.split("//", 1)[0]
        line = line.strip()
        if line:
            out.append(line)
    return "\n".join(out)


class _SafeEval(ast.NodeVisitor):
    def __init__(self, names: Dict[str, float]):
        self.names = names

    def visit_Expression(self, node: ast.Expression) -> float:
        return float(self.visit(node.body))

    def visit_Name(self, node: ast.Name) -> float:
        if node.id not in self.names:
            raise ValueError(f"Unknown symbol '{node.id}'")
        return float(self.names[node.id])

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Unsupported constant")

    def visit_Num(self, node: ast.Num) -> float:  # pragma: no cover
        return float(node.n)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        val = float(self.visit(node.operand))
        if isinstance(node.op, ast.UAdd):
            return val
        if isinstance(node.op, ast.USub):
            return -val
        raise ValueError("Unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = float(self.visit(node.left))
        right = float(self.visit(node.right))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operator")

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def eval_expr(expr: str, names: Dict[str, float]) -> float:
    tree = ast.parse(expr.strip(), mode="eval")
    return _SafeEval(names).visit(tree)


def parse_geo(path: Path) -> GeoModel:
    text = _strip_comments(path.read_text(encoding="utf-8", errors="ignore"))

    statements: List[str] = []
    current = []
    for line in text.splitlines():
        current.append(line)
        if ";" in line:
            joined = " ".join(current)
            for chunk in joined.split(";"):
                c = chunk.strip()
                if c:
                    statements.append(c)
            current = []
    if current:
        c = " ".join(current).strip()
        if c:
            statements.append(c)

    m = GeoModel(vars={"Pi": 3.141592653589793})

    assign_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
    point_re = re.compile(r"^Point\(([-+]?\d+)\)\s*=\s*\{(.+)\}$", re.IGNORECASE)
    line_re = re.compile(r"^Line\(([-+]?\d+)\)\s*=\s*\{(.+)\}$", re.IGNORECASE)
    spline_re = re.compile(r"^Spline\(([-+]?\d+)\)\s*=\s*\{(.+)\}$", re.IGNORECASE)
    curve_loop_re = re.compile(r"^(?:Curve|Line)\s+Loop\(([-+]?\d+)\)\s*=\s*\{(.+)\}$", re.IGNORECASE)
    plane_surface_re = re.compile(r"^Plane\s+Surface\(([-+]?\d+)\)\s*=\s*\{(.+)\}$", re.IGNORECASE)
    phys_curve_re = re.compile(
        r'^Physical\s+(?:Curve|Line)\("([^"]+)"(?:\s*,\s*[-+]?\d+)?\)\s*=\s*\{(.+)\}$',
        re.IGNORECASE,
    )
    phys_surface_re = re.compile(
        r'^Physical\s+Surface\("([^"]+)"(?:\s*,\s*[-+]?\d+)?\)\s*=\s*\{(.+)\}$',
        re.IGNORECASE,
    )

    for st in statements:
        mm = point_re.match(st)
        if mm:
            pid = int(mm.group(1))
            parts = [p.strip() for p in mm.group(2).split(",")]
            if len(parts) < 2:
                continue
            x = eval_expr(parts[0], m.vars)
            y = eval_expr(parts[1], m.vars)
            m.points[pid] = (x, y)
            continue

        mm = line_re.match(st)
        if mm:
            lid = int(mm.group(1))
            ids = [int(v.strip()) for v in mm.group(2).split(",") if v.strip()]
            if len(ids) >= 2:
                m.lines[lid] = (ids[0], ids[1])
            continue

        mm = spline_re.match(st)
        if mm:
            sid = int(mm.group(1))
            ids = [int(v.strip()) for v in mm.group(2).split(",") if v.strip()]
            if len(ids) >= 2:
                m.splines[sid] = ids
            continue

        mm = curve_loop_re.match(st)
        if mm:
            cid = int(mm.group(1))
            ids = [int(v.strip()) for v in mm.group(2).split(",") if v.strip()]
            m.curve_loops[cid] = ids
            continue

        mm = plane_surface_re.match(st)
        if mm:
            sid = int(mm.group(1))
            ids = [int(v.strip()) for v in mm.group(2).split(",") if v.strip()]
            m.plane_surfaces[sid] = ids
            continue

        mm = phys_curve_re.match(st)
        if mm:
            name = mm.group(1)
            ids = [int(v.strip()) for v in mm.group(2).split(",") if v.strip()]
            m.physical_curves.setdefault(name, []).extend(ids)
            continue

        mm = phys_surface_re.match(st)
        if mm:
            name = mm.group(1)
            ids = [int(v.strip()) for v in mm.group(2).split(",") if v.strip()]
            m.physical_surfaces.setdefault(name, []).extend(ids)
            continue

        mm = assign_re.match(st)
        if mm:
            key = mm.group(1)
            expr = mm.group(2)
            try:
                m.vars[key] = eval_expr(expr, m.vars)
            except Exception:
                pass

    return m


def _mesh_corner_count(element_type: int) -> int:
    # Gmsh types commonly used for 2D meshes (linear and higher-order).
    tri_types = {2, 9, 20, 21, 22, 23, 24, 25}
    quad_types = {3, 10, 16, 36, 37, 38, 39, 40, 41}
    if element_type in tri_types:
        return 3
    if element_type in quad_types:
        return 4
    return 0


def _line_corner_count(element_type: int) -> int:
    # Gmsh line element types (linear and higher order).
    line_types = {1, 8, 26, 27, 28}
    if element_type in line_types:
        return 2
    return 0


def parse_msh(path: Path) -> MshModel:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    model = MshModel()

    while i < len(lines):
        tag = lines[i].strip()

        if tag == "$PhysicalNames":
            i += 1
            if i >= len(lines):
                break
            count = int(lines[i].strip().split()[0])
            i += 1
            pn_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+"(.*)"\s*$')
            for _ in range(count):
                raw = lines[i].strip()
                mm = pn_re.match(raw)
                if mm:
                    dim = int(mm.group(1))
                    ptag = int(mm.group(2))
                    name = mm.group(3)
                    model.physical_names[ptag] = (dim, name)
                i += 1
            while i < len(lines) and lines[i].strip() != "$EndPhysicalNames":
                i += 1
            i += 1
            continue

        if tag == "$Entities":
            i += 1
            if i >= len(lines):
                break
            header = lines[i].strip().split()
            if len(header) < 4:
                raise ValueError("Invalid $Entities header in .msh")
            n_points = int(header[0])
            n_curves = int(header[1])
            n_surfaces = int(header[2])
            n_volumes = int(header[3])
            i += 1

            # Skip points.
            for _ in range(n_points):
                i += 1

            # Skip curves.
            for _ in range(n_curves):
                parts = lines[i].strip().split()
                if len(parts) >= 8:
                    ctag = int(parts[0])
                    nphys = int(parts[7])
                    pstart = 8
                    pend = pstart + nphys
                    phys_tags = [int(v) for v in parts[pstart:pend]]
                    model.curve_entity_phys[ctag] = phys_tags
                i += 1

            # Parse surfaces: tag minX minY minZ maxX maxY maxZ numPhysicalTags ...
            for _ in range(n_surfaces):
                parts = lines[i].strip().split()
                if len(parts) >= 8:
                    stag = int(parts[0])
                    nphys = int(parts[7])
                    pstart = 8
                    pend = pstart + nphys
                    phys_tags = [int(v) for v in parts[pstart:pend]]
                    model.surface_entity_phys[stag] = phys_tags
                i += 1

            # Skip volumes.
            for _ in range(n_volumes):
                i += 1

            while i < len(lines) and lines[i].strip() != "$EndEntities":
                i += 1
            i += 1
            continue

        if tag == "$Nodes":
            i += 1
            if i >= len(lines):
                break
            header = lines[i].strip().split()

            # Gmsh v2: single integer total node count.
            if len(header) == 1:
                total_nodes = int(header[0])
                i += 1
                for _ in range(total_nodes):
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        model.nodes[nid] = (x, y, z)
                    i += 1
            # Gmsh v4: numEntityBlocks numNodes minNodeTag maxNodeTag
            else:
                num_blocks = int(header[0])
                i += 1
                for _ in range(num_blocks):
                    b = lines[i].strip().split()
                    if len(b) < 4:
                        raise ValueError("Invalid $Nodes block header in .msh")
                    parametric = int(b[2])
                    count = int(b[3])
                    i += 1

                    if parametric != 0:
                        raise ValueError("Parametric nodes in .msh are not supported")

                    node_tags: List[int] = []
                    while len(node_tags) < count:
                        node_tags.extend(int(v) for v in lines[i].strip().split())
                        i += 1

                    coords: List[float] = []
                    while len(coords) < 3 * count:
                        coords.extend(float(v) for v in lines[i].strip().split())
                        i += 1

                    for k, nid in enumerate(node_tags[:count]):
                        x = coords[3 * k]
                        y = coords[3 * k + 1]
                        z = coords[3 * k + 2]
                        model.nodes[nid] = (x, y, z)

            while i < len(lines) and lines[i].strip() != "$EndNodes":
                i += 1
            i += 1
            continue

        if tag == "$Elements":
            i += 1
            if i >= len(lines):
                break
            header = lines[i].strip().split()

            # Gmsh v2: single integer total element count.
            if len(header) == 1:
                total_elems = int(header[0])
                i += 1
                for _ in range(total_elems):
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        eid = int(parts[0])
                        etype = int(parts[1])
                        ntags = int(parts[2])
                        tags = [int(v) for v in parts[3 : 3 + ntags]]
                        phys_tags = tags[:1] if tags else []
                        node_ids = [int(v) for v in parts[3 + ntags :]]
                        if _mesh_corner_count(etype) > 0:
                            model.elements.append((eid, etype, node_ids, phys_tags))
                        elif _line_corner_count(etype) > 0:
                            model.line_elements.append((eid, etype, node_ids, phys_tags))
                    i += 1
            # Gmsh v4: numEntityBlocks numElements minElementTag maxElementTag
            else:
                num_blocks = int(header[0])
                i += 1
                for _ in range(num_blocks):
                    b = lines[i].strip().split()
                    if len(b) < 4:
                        raise ValueError("Invalid $Elements block header in .msh")
                    entity_dim = int(b[0])
                    entity_tag = int(b[1])
                    etype = int(b[2])
                    count = int(b[3])
                    phys_tags = model.surface_entity_phys.get(entity_tag, []) if entity_dim == 2 else []
                    line_phys_tags = model.curve_entity_phys.get(entity_tag, []) if entity_dim == 1 else []
                    i += 1

                    for _j in range(count):
                        parts = lines[i].strip().split()
                        if parts:
                            eid = int(parts[0])
                            node_ids = [int(v) for v in parts[1:]]
                            if entity_dim == 2 and _mesh_corner_count(etype) > 0:
                                model.elements.append((eid, etype, node_ids, list(phys_tags)))
                            elif entity_dim == 1 and _line_corner_count(etype) > 0:
                                model.line_elements.append((eid, etype, node_ids, list(line_phys_tags)))
                        i += 1

            while i < len(lines) and lines[i].strip() != "$EndElements":
                i += 1
            i += 1
            continue

        i += 1

    return model


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


def _hash_color(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    r = 40 + (r % 170)
    g = 40 + (g % 170)
    b = 40 + (b % 170)
    return f"#{r:02x}{g:02x}{b:02x}"


def _pick_pattern(name: str) -> str:
    patterns = ["pat_diag", "pat_cross", "pat_dots"]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    idx = int(digest[-2:], 16) % len(patterns)
    return patterns[idx]


def _collect_curve_points(model: GeoModel, curve_ids: List[int]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for cid in curve_ids:
        seg = _entity_polyline(model, cid)
        if len(seg) < 2:
            continue
        if pts and pts[-1] == seg[0]:
            pts.extend(seg[1:])
        else:
            pts.extend(seg)
    return pts


def _curve_points_by_name(model: GeoModel, names: List[str]) -> List[List[Tuple[float, float]]]:
    out: List[List[Tuple[float, float]]] = []
    for name in names:
        ids = model.physical_curves.get(name, [])
        if not ids:
            continue
        pts = _collect_curve_points(model, ids)
        if len(pts) >= 2:
            out.append(pts)
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


@dataclass
class LegendEntry:
    names: List[str]
    label: str
    style_key: str


@dataclass
class LegendSpec:
    curve_entries: List[LegendEntry]
    surface_entries: List[LegendEntry]
    curve_style_by_name: Dict[str, str]
    surface_style_by_name: Dict[str, str]
    max_curve_items: int
    max_surface_items: int
    phreatic_line_label: Optional[str] = None
    phreatic_line_points: Optional[List[Tuple[float, float]]] = None
    legend_top_left: Tuple[float, float] = (20.0, 20.0)


def _parse_legend_entry(raw: Any, kind: str) -> Optional[Tuple[List[str], str]]:
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            return None
        return [name], name

    if not isinstance(raw, dict):
        return None

    if raw.get("show") is False:
        return None

    names: List[str] = []
    raw_names = raw.get("names")
    raw_name = raw.get("name")

    if isinstance(raw_names, list):
        names = [str(v).strip() for v in raw_names if str(v).strip()]
    elif isinstance(raw_name, str) and raw_name.strip():
        names = [raw_name.strip()]

    if not names:
        return None

    label = str(raw.get("label") or ", ".join(names)).strip()
    if not label:
        label = names[0]

    return names, label


def _legend_entries(
    names: List[str], raw_entries: Any, kind: str,
) -> Tuple[List[LegendEntry], Dict[str, str]]:
    styles = {name: name for name in names}
    if not isinstance(raw_entries, list):
        return [LegendEntry([name], name, name) for name in names], styles

    entries: List[LegendEntry] = []
    for idx, raw in enumerate(raw_entries):
        parsed = _parse_legend_entry(raw, kind)
        if not parsed:
            continue
        group_names, label = parsed
        existing = [name for name in group_names if name in styles]
        if not existing:
            continue
        style_key = f"{kind}_cfg_{idx}_{label}"
        styles.update(dict.fromkeys(existing, style_key))
        entries.append(LegendEntry(existing, label, style_key))
    return entries, styles


def _unique_group_ids(
    groups: Dict[str, List[int]], entries: List[LegendEntry],
) -> Dict[str, List[int]]:
    """Assign each geometry id to one group so shared entities are drawn once.

    Groups listed in the legend config win in config order; the remaining
    groups follow from smallest to largest, so aggregate groups lose to
    their more specific parts.
    """
    configured = [name for entry in entries for name in entry.names if name in groups]
    configured_set = set(configured)
    remaining = sorted(
        (name for name in groups if name not in configured_set),
        key=lambda name: (len(groups[name]), name),
    )
    owner: Dict[int, str] = {}
    for name in dict.fromkeys(configured + remaining):
        for gid in groups[name]:
            owner.setdefault(abs(gid), name)

    unique: Dict[str, List[int]] = {}
    for name, ids in groups.items():
        seen: set = set()
        unique[name] = []
        for gid in ids:
            if owner[abs(gid)] == name and abs(gid) not in seen:
                seen.add(abs(gid))
                unique[name].append(gid)
    return unique


def _surface_regions(
    names: List[str], style_by_name: Dict[str, str],
) -> List[Tuple[str, List[str]]]:
    """Group surface names by legend style; each style is drawn as one region."""
    regions: Dict[str, List[str]] = {}
    for name in names:
        regions.setdefault(style_by_name.get(name, name), []).append(name)
    return list(regions.items())


def _build_legend_spec(model: GeoModel, legend_cfg: Optional[Dict[str, Any]]) -> LegendSpec:
    curve_names = sorted(model.physical_curves.keys())
    surface_names = sorted(model.physical_surfaces.keys())
    legend_cfg = legend_cfg or {}
    max_curve_items = int(legend_cfg.get("max_curves", 18))
    max_surface_items = int(legend_cfg.get("max_surfaces", 18))
    curve_entries, curve_style_by_name = _legend_entries(
        curve_names, legend_cfg.get("curves"), "curve"
    )
    surface_entries, surface_style_by_name = _legend_entries(
        surface_names, legend_cfg.get("surfaces"), "surface"
    )

    # Parse phreatic line config
    phreatic_line_label: Optional[str] = None
    phreatic_line_points: Optional[List[Tuple[float, float]]] = None
    
    phreatic_cfg = legend_cfg.get("phreatic_line")
    if isinstance(phreatic_cfg, dict):
        phreatic_line_label = str(phreatic_cfg.get("label", "Phreatic Line")).strip()
        raw_points = phreatic_cfg.get("points")
        if isinstance(raw_points, list) and len(raw_points) >= 2:
            try:
                phreatic_line_points = [(float(p[0]), float(p[1])) for p in raw_points if isinstance(p, (list, tuple)) and len(p) >= 2]
                if len(phreatic_line_points) < 2:
                    phreatic_line_points = None
            except (ValueError, TypeError):
                phreatic_line_points = None
    
    # Parse legend top-left anchor in SVG coordinates.
    legend_top_left: Tuple[float, float] = (20.0, 20.0)
    raw_top_left = legend_cfg.get("top_left")
    if isinstance(raw_top_left, (list, tuple)) and len(raw_top_left) >= 2:
        try:
            legend_top_left = (float(raw_top_left[0]), float(raw_top_left[1]))
        except (ValueError, TypeError):
            legend_top_left = (20.0, 20.0)
    elif isinstance(raw_top_left, dict):
        try:
            legend_top_left = (
                float(raw_top_left.get("x", 20.0)),
                float(raw_top_left.get("y", 20.0)),
            )
        except (ValueError, TypeError):
            legend_top_left = (20.0, 20.0)
    
    return LegendSpec(
        curve_entries=curve_entries,
        surface_entries=surface_entries,
        curve_style_by_name=curve_style_by_name,
        surface_style_by_name=surface_style_by_name,
        max_curve_items=max_curve_items,
        max_surface_items=max_surface_items,
        phreatic_line_label=phreatic_line_label,
        phreatic_line_points=phreatic_line_points,
        legend_top_left=legend_top_left,
    )


def _pattern_defs(svg: ET.Element) -> None:
    defs = ET.SubElement(svg, f"{{{SVG_NS}}}defs")

    p1 = ET.SubElement(defs, f"{{{SVG_NS}}}pattern", {
        "id": "pat_diag", "patternUnits": "userSpaceOnUse", "width": "8", "height": "8"
    })
    ET.SubElement(p1, f"{{{SVG_NS}}}path", {
        "d": "M 0 8 L 8 0", "stroke": "#444", "stroke-width": "1", "fill": "none"
    })

    p2 = ET.SubElement(defs, f"{{{SVG_NS}}}pattern", {
        "id": "pat_cross", "patternUnits": "userSpaceOnUse", "width": "8", "height": "8"
    })
    ET.SubElement(p2, f"{{{SVG_NS}}}path", {
        "d": "M 0 8 L 8 0 M 0 0 L 8 8", "stroke": "#444", "stroke-width": "1", "fill": "none"
    })

    p3 = ET.SubElement(defs, f"{{{SVG_NS}}}pattern", {
        "id": "pat_dots", "patternUnits": "userSpaceOnUse", "width": "8", "height": "8"
    })
    ET.SubElement(p3, f"{{{SVG_NS}}}circle", {
        "cx": "2", "cy": "2", "r": "1.2", "fill": "#444"
    })


Point = Tuple[float, float]
Segment = Tuple[Point, Point]


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


def _new_svg(width: int, height: int) -> ET.Element:
    svg = ET.Element(
        f"{{{SVG_NS}}}svg",
        {"width": str(width), "height": str(height), "viewBox": f"0 0 {width} {height}"},
    )
    ET.SubElement(svg, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "#ffffff"
    })
    return svg


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
        color = _hash_color(style_key)
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
            "stroke": _hash_color(style_key),
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

    # Place legend using explicit top-left coordinates from config.
    lx = float(legend_spec.legend_top_left[0])
    ly = float(legend_spec.legend_top_left[1])
    lx2 = lx + 360.0
    
    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx), "y": str(ly), "font-size": "14", "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Physical Curves (conditions)"

    y = ly + 18
    curve_legend = legend_spec.curve_entries[: max(0, legend_spec.max_curve_items)]
    for entry in curve_legend:
        color = _hash_color(entry.style_key)
        ET.SubElement(g_legend, f"{{{SVG_NS}}}line", {
            "x1": str(lx), "y1": str(y), "x2": str(lx + 18), "y2": str(y),
            "stroke": color, "stroke-width": "3"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx + 24), "y": str(y + 4), "font-size": "11", "font-family": "DejaVu Sans", "fill": "#111"
        }).text = entry.label
        y += 16
    
    # Draw phreatic line in legend if provided
    if legend_spec.phreatic_line_label and legend_spec.phreatic_line_points:
        y += 8  # Add extra space before phreatic line
        ET.SubElement(g_legend, f"{{{SVG_NS}}}line", {
            "x1": str(lx), "y1": str(y - 4), "x2": str(lx + 18), "y2": str(y - 4),
            "stroke": "#0066cc", "stroke-width": "2", "stroke-dasharray": "5,3"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx + 24), "y": str(y), "font-size": "11", "font-family": "DejaVu Sans", "fill": "#111"
        }).text = legend_spec.phreatic_line_label

    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx2), "y": str(ly), "font-size": "14", "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Physical Surfaces (elements)"

    y2 = ly + 18
    surface_legend = legend_spec.surface_entries[: max(0, legend_spec.max_surface_items)]
    for entry in surface_legend:
        color = _hash_color(entry.style_key)
        pat = _pick_pattern(entry.style_key)
        ET.SubElement(g_legend, f"{{{SVG_NS}}}rect", {
            "x": str(lx2), "y": str(y2 - 10), "width": "16", "height": "12",
            "fill": color, "fill-opacity": "0.25", "stroke": color, "stroke-width": "1"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}rect", {
            "x": str(lx2), "y": str(y2 - 10), "width": "16", "height": "12",
            "fill": f"url(#{pat})", "fill-opacity": "0.45", "stroke": "none"
        })
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx2 + 24), "y": str(y2), "font-size": "11", "font-family": "DejaVu Sans", "fill": "#111"
        }).text = entry.label
        y2 += 16

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
        ccolor = _hash_color(style_key)
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
        elem_color[eid] = _hash_color(style_key)

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

    lx = 20
    ly = 24
    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx), "y": str(ly), "font-size": "14", "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Mesh view (.msh input)"
    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx), "y": str(ly + 18), "font-size": "11", "font-family": "DejaVu Sans", "fill": "#111"
    }).text = f"Elements rendered: {len(polygons)} (line-only view)"
    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx), "y": str(ly + 34), "font-size": "11", "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Edges: surface style colors"
    ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
        "x": str(lx), "y": str(ly + 50), "font-size": "11", "font-family": "DejaVu Sans", "fill": "#111"
    }).text = "Boundary conditions: curve style colors"
    if shared_edges_black:
        ET.SubElement(g_legend, f"{{{SVG_NS}}}text", {
            "x": str(lx), "y": str(ly + 66), "font-size": "11", "font-family": "DejaVu Sans", "fill": "#111"
        }).text = "Interfaces between surface styles: black"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(out_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GEO/MSH/MDPA file to styled SVG.")
    parser.add_argument("--input", required=True, help="Path to .geo, .msh or .mdpa file")
    parser.add_argument("--output", required=True, help="Path to .svg output")
    parser.add_argument("--width", type=int, default=2200, help="SVG width in px")
    parser.add_argument("--height", type=int, default=1200, help="SVG height in px")
    parser.add_argument("--padding", type=int, default=70, help="Padding in px")
    parser.add_argument(
        "--legend-config",
        help="Optional JSON file with legend entries and grouped names",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    legend_cfg: Optional[Dict[str, Any]] = None
    if args.legend_config:
        cfg_path = Path(args.legend_config)
        legend_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(legend_cfg, dict):
            raise ValueError("Legend config root must be a JSON object")
        legend_cfg = legend_cfg.get("legend") if "legend" in legend_cfg else legend_cfg
        if not isinstance(legend_cfg, dict):
            raise ValueError("Legend config must contain a JSON object")

    suffix = in_path.suffix.lower()

    if suffix in {".geo", ".mdpa"}:
        model = parse_mdpa(in_path) if suffix == ".mdpa" else parse_geo(in_path)
        geo_to_svg(
            model,
            out_path,
            width=args.width,
            height=args.height,
            padding=args.padding,
            legend_cfg=legend_cfg,
        )

        print(f"Wrote SVG: {out_path}")
        print(
            f"Parsed points={len(model.points)}, lines={len(model.lines)}, "
            f"splines={len(model.splines)}, loops={len(model.curve_loops)}"
        )
        print(
            f"Physical curves={len(model.physical_curves)}, "
            f"physical surfaces={len(model.physical_surfaces)}"
        )
    elif suffix == ".msh":
        msh_model = parse_msh(in_path)
        msh_to_svg(
            msh_model,
            out_path,
            width=args.width,
            height=args.height,
            padding=args.padding,
            legend_cfg=legend_cfg,
        )
        print(f"Wrote SVG: {out_path}")
        print(f"Parsed nodes={len(msh_model.nodes)}, elements={len(msh_model.elements)}")
    else:
        raise ValueError("Unsupported input extension. Use .geo, .msh or .mdpa")


if __name__ == "__main__":
    main()
