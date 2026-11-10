"""Kratos .mdpa adapter producing a GeoModel."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..mdpa import _connection_paths, parse_mdpa_data
from .models import GeoModel


def _mdpa_memberships(
    path: Path,
) -> Tuple[Dict[Tuple[str, int], set], Dict[str, set]]:
    """Return SubModelPart entity and node memberships keyed by nested name.

    Nested SubModelParts are named ``Parent/Child``; an entity or node belongs
    to the SubModelPart that lists it.
    """
    entity_groups: Dict[Tuple[str, int], set] = {}
    node_groups: Dict[str, set] = {}
    stack: List[str] = []
    listing: Optional[str] = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("//", 1)[0].strip()
            if not line:
                continue
            words = line.split()
            keyword = " ".join(words[:2]).lower()
            if keyword == "begin submodelpart" and len(words) >= 3:
                stack.append(" ".join(words[2:]))
                listing = None
            elif keyword == "end submodelpart":
                if stack:
                    stack.pop()
                listing = None
            elif stack and keyword in {
                "begin submodelpartnodes",
                "begin submodelpartelements",
                "begin submodelpartconditions",
            }:
                listing = words[1][len("SubModelPart"):].capitalize()
                if listing == "Nodes":
                    node_groups.setdefault("/".join(stack), set())
            elif words[0].lower() == "end":
                listing = None
            elif listing and stack:
                try:
                    item_id = int(words[0])
                except ValueError:
                    continue
                name = "/".join(stack)
                if listing == "Nodes":
                    node_groups[name].add(item_id)
                else:
                    entity_groups.setdefault((listing, item_id), set()).add(name)
    return entity_groups, node_groups


def parse_mdpa(path: Path) -> GeoModel:
    """Adapt supported 2D MDPA geometry to named SVG surfaces and curves.

    Node-only SubModelParts become curves along the outer element edges whose
    two end nodes both belong to the group.
    """
    parsed = parse_mdpa_data(path)
    if not parsed.nodes:
        raise ValueError("No nodes parsed from MDPA file.")
    if not parsed.entity_records:
        raise ValueError("No elements or conditions parsed from MDPA file.")
    entity_groups, node_groups = _mdpa_memberships(path)

    model = GeoModel(points={nid: (x, y) for nid, (x, y, _z) in parsed.nodes.items()})
    surface_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for block, entity_type, entity_id, node_ids in parsed.entity_records:
        missing = sorted(set(node_ids) - model.points.keys())
        if missing:
            raise ValueError(f"{block} {entity_id} references missing nodes: {missing}")
        try:
            paths = _connection_paths(entity_type, node_ids)
        except ValueError as error:
            raise ValueError(f"{block} {entity_id}: {error}") from error
        names = sorted(entity_groups.get((block, entity_id), ())) or [entity_type]
        for node_path in paths:
            curve_id = len(model.splines) + 1
            model.splines[curve_id] = list(node_path)
            if len(node_path) >= 4 and node_path[0] == node_path[-1]:
                surface_id = len(model.plane_surfaces) + 1
                model.curve_loops[surface_id] = [curve_id]
                model.plane_surfaces[surface_id] = [surface_id]
                for name in names:
                    model.physical_surfaces.setdefault(name, []).append(surface_id)
                for edge in zip(node_path, node_path[1:]):
                    surface_edges.setdefault(tuple(sorted(edge)), []).append(edge)
            else:
                for name in names:
                    model.physical_curves.setdefault(name, []).append(curve_id)

    entity_names = {name for names in entity_groups.values() for name in names}
    outer_edges = [edges[0] for edges in surface_edges.values() if len(edges) == 1]
    edge_curve_ids: Dict[Tuple[int, int], int] = {}
    for name in sorted(set(node_groups) - entity_names):
        nodes = node_groups[name]
        for edge in outer_edges:
            if edge[0] not in nodes or edge[1] not in nodes:
                continue
            if edge not in edge_curve_ids:
                edge_curve_ids[edge] = len(model.splines) + 1
                model.splines[edge_curve_ids[edge]] = list(edge)
            model.physical_curves.setdefault(name, []).append(edge_curve_ids[edge])
    return model
