"""Gmsh .geo reader with a safe arithmetic expression evaluator."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List

from .models import GeoModel


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
