"""Gmsh .msh (v2/v4) reader."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .models import MshModel


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
