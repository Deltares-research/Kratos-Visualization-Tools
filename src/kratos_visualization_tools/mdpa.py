"""MDPA file parsing and mesh topology helpers."""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

def _normalize_submodelpart_name(model_part_name):
    """Return the leaf SubModelPart name from a model part path."""
    if not isinstance(model_part_name, str):
        return ''
    return model_part_name.split('.')[-1]


def _stage_from_process_label(label):
    """Extract stage name from labels like '[stage] Process', else 'Global'."""
    if isinstance(label, str) and label.startswith('['):
        end = label.find(']')
        if end > 1:
            return label[1:end]
    return 'Global'


def _connection_paths(entity_type, node_ids):
    """Return ordered polylines for a Kratos element or condition."""
    type_name = (entity_type or '').lower()
    if '3plus3n' in type_name and len(node_ids) == 6:
        return [(node_ids[0], node_ids[2], node_ids[1]),
                (node_ids[3], node_ids[5], node_ids[4])]
    if '2plus2n' in type_name and len(node_ids) == 4:
        return [(node_ids[0], node_ids[1]), (node_ids[2], node_ids[3])]
    if '2d6n' in type_name and len(node_ids) == 6:
        return [(node_ids[0], node_ids[3], node_ids[1],
                 node_ids[4], node_ids[2], node_ids[5], node_ids[0])]
    if '2d8n' in type_name and len(node_ids) == 8:
        return [(node_ids[0], node_ids[4], node_ids[1],
                 node_ids[5], node_ids[2], node_ids[6],
                 node_ids[3], node_ids[7], node_ids[0])]
    if '2d3n' in type_name and len(node_ids) == 3:
        return [(node_ids[0], node_ids[2], node_ids[1])]
    return [node_ids]


def _normal_segments(entity_type, node_ids):
    """Return quadratic/linear boundary segments used for normal vectors."""
    paths = _connection_paths(entity_type, node_ids)
    segments = []
    for path in paths:
        if len(path) == 2:
            segments.append((path[0], path[1]))
        elif len(path) == 3:
            segments.append((path[0], path[1], path[2]))
    return segments


@dataclass
class ParsedMdpa:
    """Structured MDPA data used by the interactive visualizer."""

    nodes: dict
    submodel_nodes: dict
    connections: list
    entity_submodels: dict
    entity_records: list


def _normal_vector(start, end, length):
    """Return a left-hand 2D normal vector of the requested length."""
    tangent = np.array([end[0] - start[0], end[1] - start[1]])
    tangent_length = np.linalg.norm(tangent)
    if tangent_length == 0:
        return None
    normal = np.array([-tangent[1], tangent[0]])
    return normal / tangent_length * length


def parse_mdpa_data(mdpa_path):
    """Parse an MDPA file into the structured visualizer data model."""
    nodes, submodel_nodes, connections, entity_submodels, entity_records = parse_mdpa(
        mdpa_path, include_connections=True
    )
    return ParsedMdpa(
        nodes=nodes,
        submodel_nodes=submodel_nodes,
        connections=connections,
        entity_submodels=entity_submodels,
        entity_records=entity_records,
    )


def parse_mdpa(mdpa_path, include_connections=False):
    """Parse Kratos .mdpa nodes, SubModelParts, and optional mesh connections."""
    nodes = {}  # id -> (x,y,z)
    submodel_nodes = defaultdict(set)  # name -> set(node_ids)
    connections = []
    entity_records = []
    entity_submodels = defaultdict(set)

    current_sub = None
    in_nodes = False
    in_sub_nodes = False
    in_sub_entities = None
    current_entity_block = None
    current_entity_type = None

    with open(mdpa_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            line_lower = line.lower()
            
            # Main model nodes block
            if line_lower.startswith('begin nodes'):
                in_nodes = True
                continue
            if line_lower.startswith('end nodes'):
                in_nodes = False
                continue

            if in_nodes:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        nid = int(parts[0])
                        x, y, z = map(float, parts[1:4])
                        nodes[nid] = (x, y, z)
                    except ValueError:
                        pass
                continue

            if line_lower.startswith('begin elements'):
                current_entity_block = 'Elements'
                current_entity_type = line.split()[2]
                continue
            if line_lower.startswith('begin conditions'):
                current_entity_block = 'Conditions'
                current_entity_type = line.split()[2]
                continue
            if line_lower.startswith('end elements') or line_lower.startswith('end conditions'):
                current_entity_block = None
                current_entity_type = None
                continue

            if current_entity_block:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        node_ids = tuple(int(value) for value in parts[2:])
                    except ValueError:
                        node_ids = ()
                    if node_ids:
                        connections.append((current_entity_type, node_ids))
                        entity_records.append(
                            (current_entity_block, current_entity_type,
                             int(parts[0]), node_ids)
                        )
                continue

            # SubModelPart block - must distinguish from SubModelPartNodes
            if line_lower.startswith('begin submodelpart ') and not line_lower.startswith('begin submodelpartnodes'):
                parts = line.split()
                if len(parts) >= 3:
                    current_sub = ' '.join(parts[2:]).split('//', 1)[0].rstrip()
                else:
                    current_sub = 'Unnamed'
                in_sub_nodes = False
                in_sub_entities = None
                continue

            # SubModelPartNodes block (within a SubModelPart)
            if current_sub and line_lower.startswith('begin submodelpartnodes'):
                in_sub_nodes = True
                continue
            if line_lower.startswith('end submodelpartnodes'):
                in_sub_nodes = False
                continue

            if current_sub and line_lower.startswith('begin submodelpartelements'):
                in_sub_entities = 'Elements'
                continue
            if current_sub and line_lower.startswith('begin submodelpartconditions'):
                in_sub_entities = 'Conditions'
                continue
            if line_lower.startswith('end submodelpartelements') or line_lower.startswith('end submodelpartconditions'):
                in_sub_entities = None
                continue

            # Inside SubModelPartNodes: each line is one node ID
            if in_sub_nodes and current_sub:
                try:
                    nid = int(line)
                    submodel_nodes[current_sub].add(nid)
                except ValueError:
                    pass

            if in_sub_entities and current_sub:
                try:
                    entity_id = int(line.split()[0])
                    entity_submodels[(in_sub_entities, entity_id)].add(current_sub)
                except (IndexError, ValueError):
                    pass

            if line_lower.startswith('end submodelpart'):
                current_sub = None
                in_sub_nodes = False
                in_sub_entities = None
                continue

    if include_connections:
        return nodes, submodel_nodes, connections, entity_submodels, entity_records
    return nodes, submodel_nodes


