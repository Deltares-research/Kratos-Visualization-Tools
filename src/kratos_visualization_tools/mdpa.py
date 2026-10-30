"""MDPA file parsing, mesh topology, and display-only face separation."""
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import product

import numpy as np


class GeometryType(Enum):
    LINE_2D3 = "Line2D3"
    TRIANGLE_2D3 = "Triangle2D3"


ENTITY_2D3N_GEOMETRIES = {
    "LinearTimoshenkoCurvedBeamElement2D3N": GeometryType.LINE_2D3,
    "LinearTimoshenkoBeamElement2D3N": GeometryType.LINE_2D3,
    "LineLoadDiffOrderCondition2D3N": GeometryType.LINE_2D3,
    "LineNormalLoadDiffOrderCondition2D3N": GeometryType.LINE_2D3,
    "LineCondition2D3N": GeometryType.LINE_2D3,
    "TransientPwElement2D3N": GeometryType.TRIANGLE_2D3,
    "SteadyStatePwElement2D3N": GeometryType.TRIANGLE_2D3,
    "TransientPwLineElement2D3N": GeometryType.LINE_2D3,
    "UPwSmallStrainElement2D3N": GeometryType.TRIANGLE_2D3,
    "UPwSmallStrainFICElement2D3N": GeometryType.TRIANGLE_2D3,
    "UPwSmallStrainAxisymmetricElement2D3N": GeometryType.TRIANGLE_2D3,
    "UPwSmallStrainAxisymmetricFICElement2D3N": GeometryType.TRIANGLE_2D3,
    "UPwUpdatedLagrangianElement2D3N": GeometryType.TRIANGLE_2D3,
    "UPwUpdatedLagrangianFICElement2D3N": GeometryType.TRIANGLE_2D3,
    "UPwUpdatedLagrangianAxisymmetricElement2D3N": GeometryType.TRIANGLE_2D3,
    "UPwUpdatedLagrangianAxisymmetricFICElement2D3N": GeometryType.TRIANGLE_2D3,
    "GeoTransientThermalElement2D3N": GeometryType.TRIANGLE_2D3,
    "GeoTransientThermalLineElement2D3N": GeometryType.LINE_2D3,
    "Element2D3N": GeometryType.TRIANGLE_2D3,
    "UPwNormalFaceLoadCondition2D3N": GeometryType.LINE_2D3,
}


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


def _resolve_constraint_relations(relations, submodel_nodes):
    """Resolve qualified constraint paths to the most-qualified MDPA keys."""
    resolved = []
    for master, slave, stage in relations:
        names = []
        for name in (master, slave):
            candidate = name
            while candidate not in submodel_nodes and '.' in candidate:
                candidate = candidate.split('.', 1)[1]
            if candidate not in submodel_nodes:
                print(f"Warning: Could not resolve constraint SubModelPart '{name}' in MDPA.")
                break
            names.append(candidate)
        if len(names) == 2:
            relationship = (names[0], names[1], stage)
            if relationship not in resolved:
                resolved.append(relationship)
    return resolved


def _connection_paths(entity_type, node_ids):
    """Return ordered polylines for a Kratos element or condition."""
    type_name = (entity_type or '').lower()
    if '3plus3n' in type_name and len(node_ids) == 6:
        return [(node_ids[0], node_ids[2], node_ids[1]),
                (node_ids[3], node_ids[5], node_ids[4])]
    if '2plus2n' in type_name and len(node_ids) == 4:
        return [(node_ids[0], node_ids[1]), (node_ids[2], node_ids[3])]
    if '2d2n' in type_name and len(node_ids) == 2:
        return [(node_ids[0], node_ids[1])]
    if '2d3n' in type_name and len(node_ids) == 3:
        geometry = next(
            (geometry for name, geometry in ENTITY_2D3N_GEOMETRIES.items()
             if name.lower() == type_name),
            None,
        )
        if geometry is GeometryType.LINE_2D3:
            return [(node_ids[0], node_ids[2], node_ids[1])]
        if geometry is GeometryType.TRIANGLE_2D3:
            return [(node_ids[0], node_ids[1], node_ids[2], node_ids[0])]
        raise ValueError(f"Unknown geometry for entity type: {entity_type}")
    if '2d4n' in type_name and len(node_ids) == 4:
        return [(node_ids[0], node_ids[1], node_ids[2], node_ids[3], node_ids[0])]
    if '2d6n' in type_name and len(node_ids) == 6:
        return [(node_ids[0], node_ids[3], node_ids[1],
                 node_ids[4], node_ids[2], node_ids[5], node_ids[0])]
    if '2d8n' in type_name and len(node_ids) == 8:
        return [(node_ids[0], node_ids[4], node_ids[1],
                 node_ids[5], node_ids[2], node_ids[6],
                 node_ids[3], node_ids[7], node_ids[0])]
    raise ValueError(f"Unknown geometry type: {type_name} with a number of nodes of {len(node_ids)}")


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


def _separation_paths(block, entity_type, entity_id, node_ids, issues):
    """Skip unsupported diagnostic topology without disabling the node view."""
    try:
        return _connection_paths(entity_type, node_ids)
    except ValueError as error:
        diagnostic = f'Unsupported separation geometry: {block} {entity_id} ({entity_type}): {error}'
        for node_id in node_ids:
            issues.setdefault(node_id, set()).add(diagnostic)
        return []


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


def _build_face_separation(parsed, constraint_relations=(), tolerance=1.0e-8,
                           visible_interfaces=None):
    """Resolve display offsets from compatible visible interface contributions.

    Interface keys are (block, entity_id); None includes every interface.
    Opening directions more than 90 degrees apart leave the shared duplicate fixed.
    """
    constraint_relations = _resolve_constraint_relations(
        constraint_relations, parsed.submodel_nodes)
    incident = defaultdict(list)
    for record in parsed.entity_records:
        for node_id in record[3]:
            incident[node_id].append(record)
    pairs = {}
    contributions = defaultdict(dict)
    issues = defaultdict(set)
    master_nodes = set()
    slave_nodes = set()
    constraint_slaves = set()
    master_candidates = defaultdict(set)
    ambiguous_slaves = set()
    for master_name, slave_name, stage in constraint_relations:
        master_nodes.update(parsed.submodel_nodes.get(master_name, set()))
        slave_nodes.update(parsed.submodel_nodes.get(slave_name, set()))

    def register(first, second):
        if first == second or first not in parsed.nodes or second not in parsed.nodes:
            return None
        if np.linalg.norm(np.array(parsed.nodes[first]) - parsed.nodes[second]) > tolerance:
            return None
        key = tuple(sorted((first, second)))
        return pairs.setdefault(key, {'interfaces': set(), 'constraints': set()})

    for block, entity_type, entity_id, node_ids in parsed.entity_records:
        if not any(name in entity_type.lower() for name in ('2plus2n', '3plus3n')):
            continue
        if visible_interfaces is not None and (block, entity_id) not in visible_interfaces:
            continue
        paths = _separation_paths(block, entity_type, entity_id, node_ids, issues)
        if not paths:
            continue
        first_path, second_path = paths
        start = parsed.nodes[first_path[0]][:2]
        end = parsed.nodes[first_path[-1]][:2]
        normal = _normal_vector(start, end, 1.0)
        if normal is None:
            continue
        if tuple(start) > tuple(end):
            normal = -normal
        for first in first_path:
            candidates = [second for second in second_path if second != first and
                          np.linalg.norm(np.array(parsed.nodes[first]) - parsed.nodes[second]) <= tolerance]
            if len(candidates) != 1:
                issues[first].add('Ambiguous or noncoincident interface counterpart')
                continue
            second = candidates[0]
            relationship = register(first, second)
            if relationship is None:
                continue
            reference, duplicate = sorted((first, second))
            if duplicate in master_nodes and reference in slave_nodes:
                reference, duplicate = duplicate, reference
            relationship['interfaces'].add((block, entity_id))
            relationship.setdefault('moving_nodes', set()).add(duplicate)
            direction = normal.copy()
            if len(first_path) == 3:
                first_point, mid_point, last_point = [np.array(parsed.nodes[node][:2]) for node in first_path]
                tangent = {
                    first_path[0]: -3 * first_point + 4 * mid_point - last_point,
                    first_path[1]: last_point - first_point,
                    first_path[2]: first_point - 4 * mid_point + 3 * last_point,
                }[first]
                local_normal = _normal_vector((0, 0), tangent, 1.0)
                if local_normal is not None:
                    direction = -local_normal if tuple(start) > tuple(end) else local_normal
            duplicate_path = first_path if duplicate in first_path else second_path
            face_neighbors = set(duplicate_path) - {duplicate}
            centers = []
            for neighbor in incident[duplicate]:
                # A shared tip can touch soil on both sides of the structure.
                # Orient this interface using only soil attached to its own face.
                if neighbor[0] != 'Elements' or not face_neighbors.intersection(neighbor[3]):
                    continue
                if '2d6n' in neighbor[1].lower() or ('2d3n' in neighbor[1].lower() and len(neighbor[3]) == 3):
                    if 'line' not in neighbor[1].lower() and 'beam' not in neighbor[1].lower():
                        centers.append(np.mean([parsed.nodes[node][:2] for node in neighbor[3][:3]], axis=0))
                elif any(name in neighbor[1].lower() for name in ('2d4n', '2d8n')):
                    centers.append(np.mean([parsed.nodes[node][:2] for node in neighbor[3][:4]], axis=0))
            if centers and np.dot(np.mean(centers, axis=0) - parsed.nodes[duplicate][:2], direction) < 0:
                direction = -direction
            contributions[duplicate][(block, entity_id)] = direction

    for master_name, slave_name, stage in constraint_relations:
        masters = parsed.submodel_nodes.get(master_name, set())
        slaves = parsed.submodel_nodes.get(slave_name, set())
        coordinates = defaultdict(list)
        for master in masters:
            if master in parsed.nodes:
                coordinates[tuple(np.floor(np.array(parsed.nodes[master]) / tolerance).astype(int))].append(master)
        for slave in slaves:
            if slave not in parsed.nodes:
                continue
            bucket = np.floor(np.array(parsed.nodes[slave]) / tolerance).astype(int)
            matches = []
            for offset in product((-1, 0, 1), repeat=3):
                for master in coordinates.get(tuple(bucket + offset), []):
                    if master != slave and np.linalg.norm(np.array(parsed.nodes[master]) - parsed.nodes[slave]) <= tolerance:
                        matches.append(master)
            for master in matches:
                relationship = register(master, slave)
                if relationship is not None:
                    relationship['constraints'].add((stage, master_name, slave_name))
                    constraint_slaves.add(slave)
                    master_candidates[slave].add(master)

    for slave, candidates in master_candidates.items():
        if len(candidates) > 1:
            issues[slave].add('Multiple coincident master candidates')
            ambiguous_slaves.add(slave)

    role_conflicts = master_nodes & slave_nodes
    for node_id in role_conflicts:
        issues[node_id].add('Conflicting master/slave roles; node left fixed')
    fixed_nodes = master_nodes | ambiguous_slaves
    moving_nodes = constraint_slaves | {
        node_id for relationship in pairs.values()
        for node_id in relationship.get('moving_nodes', ())
    }
    moving_nodes.difference_update(fixed_nodes)

    for duplicate in sorted(moving_nodes):
        if duplicate in contributions:
            continue
        for block, entity_type, entity_id, node_ids in incident[duplicate]:
            if '2plus2n' in entity_type.lower() or '3plus3n' in entity_type.lower():
                continue
            if block != 'Conditions' and not any(name in entity_type.lower() for name in ('line', 'beam')):
                continue
            for path in _separation_paths(block, entity_type, entity_id, node_ids, issues):
                if len(path) not in (2, 3):
                    continue
                normal = _normal_vector(parsed.nodes[path[0]], parsed.nodes[path[-1]], 1.0)
                if normal is not None:
                    if parsed.nodes[path[0]][:2] > parsed.nodes[path[-1]][:2]:
                        normal = -normal
                    contributions[duplicate][(block, entity_id)] = normal

    interface_keys = {key for relationship in pairs.values()
                      for key in relationship['interfaces']}
    directions = {}
    conflicting_nodes = set()
    for duplicate in sorted(moving_nodes):
        guides = list(contributions.get(duplicate, {}).values())
        interface_guides = [
            direction for entity_key, direction in contributions.get(duplicate, {}).items()
            if entity_key in interface_keys
        ]
        if any(np.dot(first, second) < -1.0e-8
               for index, first in enumerate(interface_guides)
               for second in interface_guides[index + 1:]):
            conflicting_nodes.add(duplicate)
            issues[duplicate].add('Conflicting visible interface directions; node left fixed')
            continue
        neighbors = {}
        for block, entity_type, entity_id, node_ids in incident[duplicate]:
            type_name = entity_type.lower()
            is_interface = '2plus2n' in type_name or '3plus3n' in type_name
            if is_interface and visible_interfaces is not None and (block, entity_id) not in visible_interfaces:
                continue
            paths = _separation_paths(block, entity_type, entity_id, node_ids, issues)
            if not paths:
                continue
            if (block == 'Elements' and '2d3n' in type_name and len(node_ids) == 3
                    and not any(name in type_name for name in ('line', 'beam'))):
                paths = [tuple(node_ids) + (node_ids[0],)]
            for path in paths:
                for first, second in zip(path, path[1:]):
                    if duplicate not in (first, second):
                        continue
                    neighbor = second if first == duplicate else first
                    if neighbor not in parsed.nodes:
                        continue
                    vector = np.array(parsed.nodes[neighbor][:2]) - parsed.nodes[duplicate][:2]
                    magnitude = np.linalg.norm(vector)
                    if magnitude > tolerance:
                        previous = neighbors.get(neighbor)
                        non_interface = not is_interface or (previous is not None and previous[0])
                        neighbors[neighbor] = (non_interface, vector / magnitude)
        if not neighbors:
            issues[duplicate].add('No noncoincident connected edge available; node left fixed')
            continue
        selected_directions = []
        for guide in guides or [np.zeros(2)]:
            neighbor = max(neighbors, key=lambda node_id: (
                neighbors[node_id][0], round(float(np.dot(neighbors[node_id][1], guide)), 12), -node_id))
            selected_directions.append(neighbors[neighbor][1])
        if interface_guides and any(
                np.dot(first, second) < -1.0e-8
                for index, first in enumerate(selected_directions)
                for second in selected_directions[index + 1:]):
            conflicting_nodes.add(duplicate)
            issues[duplicate].add('Conflicting visible interface directions; node left fixed')
            continue
        directions[duplicate] = np.mean(selected_directions, axis=0)

    midside_corners = defaultdict(set)
    for block, entity_type, entity_id, node_ids in parsed.entity_records:
        type_name = entity_type.lower()
        is_interface = '2plus2n' in type_name or '3plus3n' in type_name
        if is_interface and visible_interfaces is not None and (block, entity_id) not in visible_interfaces:
            continue
        is_quadratic = any(name in type_name for name in ('3plus3n', '2d6n', '2d8n'))
        is_quadratic_line = len(node_ids) == 3 and (
            any(name in type_name for name in ('line', 'beam'))
            or (block == 'Conditions' and '2d3n' in type_name))
        if not (is_quadratic or is_quadratic_line):
            continue
        for path in _separation_paths(block, entity_type, entity_id, node_ids, issues):
            for index in range(0, len(path) - 2, 2):
                first, midside, second = path[index:index + 3]
                if all(node_id in parsed.nodes for node_id in (first, midside, second)):
                    midside_corners[midside].add(tuple(sorted((first, second))))
    for midside in midside_corners:
        directions.pop(midside, None)
    counts = defaultdict(int)
    for node_id in sorted(directions):
        if not np.any(directions[node_id]):
            continue
        counts[parsed.nodes[node_id]] += 1
        directions[node_id] = directions[node_id] * counts[parsed.nodes[node_id]]
    corner_directions = dict(directions)
    for midside, edges in midside_corners.items():
        if midside in conflicting_nodes or midside in fixed_nodes:
            continue
        directions[midside] = np.mean([
            (np.array(corner_directions.get(first, (0, 0)))
             + corner_directions.get(second, (0, 0))) / 2
            for first, second in sorted(edges)
        ], axis=0)
        if midside in issues and any(
                first in corner_directions or second in corner_directions
                for first, second in edges):
            issues[midside].discard('No noncoincident connected edge available; node left fixed')
            if not issues[midside]:
                del issues[midside]
    for node_id in conflicting_nodes:
        directions[node_id] = np.zeros(2)
    for node_id in fixed_nodes:
        directions.pop(node_id, None)
    return incident, pairs, directions, dict(issues)


def _separated_coordinates(nodes, normals, gap):
    """Scale resolved rank-aware displacement vectors by one hundredth of gap."""
    offset_gap = 0.01 * gap
    return {node_id: (position[0] + offset_gap * normals.get(node_id, (0, 0))[0],
                      position[1] + offset_gap * normals.get(node_id, (0, 0))[1], position[2])
            for node_id, position in nodes.items()}


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
            if line_lower.startswith('begin submodelpart '):
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

            if line_lower.split('//', 1)[0].strip() == 'end submodelpart':
                current_sub = None
                in_sub_nodes = False
                in_sub_entities = None
                continue

    if include_connections:
        return nodes, submodel_nodes, connections, entity_submodels, entity_records
    return nodes, submodel_nodes
