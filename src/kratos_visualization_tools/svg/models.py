"""GeoModel, MSH models and shared geometry type aliases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


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


Point = Tuple[float, float]
Segment = Tuple[Point, Point]
