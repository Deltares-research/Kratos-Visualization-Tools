"""GiD mesh, result metadata, and viewer state models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


RESULT_HEADER_RE = re.compile(
    r'^Result\s+"(?P<name>[^"]+)"\s+"(?P<source>[^"]+)"\s+(?P<step>\S+)\s+(?P<rtype>\S+)\s+(?P<loc>\S+)(?:\s+"(?P<gp>[^"]+)")?\s*$'
)


MESH_HEADER_RE = re.compile(
    r'^MESH\s+"(?P<name>[^"]+)"\s+dimension\s+(?P<dim>\d+)\s+ElemType\s+(?P<etype>\S+)\s+Nnode\s+(?P<nnode>\d+)\s*$'
)


RESULT_LOCATION_ON_NODES = "OnNodes"


RESULT_LOCATION_ON_GAUSS_POINTS = "OnGaussPoints"


VECTOR_COMPONENTS = ("x", "y", "z", "magnitude")


TENSOR_COMPONENTS = ("xx", "xy", "yx", "yy", "trace", "magnitude")


@dataclass
class MeshBlock:
    name: str
    elem_type: str
    nnode: int
    coordinates: Dict[int, Tuple[float, float, float]]
    elements: List[Tuple[int, List[int], int]]


@dataclass
class ResultHeader:
    name: str
    step: int
    result_type: str
    location: str
    gauss_points: Optional[str]
    header_line: int


@dataclass
class ActiveSubmodelPart:
    name: str
    node_ids: set
    entity_ids: set


@dataclass
class SolutionData:
    label: str
    msh_path: Path
    res_path: Path
    headers: List[ResultHeader]
    all_nodes: Dict[int, Tuple[float, float, float]]
    triangles: List[Tuple[int, int, int]]
    surface_elements: List[Tuple[int, List[int]]]
    line_meshes: list
    profile_nodes: set


@dataclass
class ViewState:
    result: str
    component: str = "magnitude"
    tensor_component: str = "trace"
    show_gradient: bool = False
    show_mesh: bool = False
    show_isolines: bool = False
    show_isoline_labels: bool = False
    show_profile: bool = False
