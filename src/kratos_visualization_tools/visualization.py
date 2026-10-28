"""Interactive matplotlib visualization and user interface."""
import colorsys
import json
import os
import re
import subprocess
import sys
import textwrap
import warnings
from datetime import datetime
from time import perf_counter
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
import numpy as np

from .mdpa import (
    _connection_paths,
    _normalize_submodelpart_name,
    _normal_segments,
    _normal_vector,
    _stage_from_process_label,
    parse_mdpa_data,
)
from .project import parse_project_parameters

def visualize_mdpa(mdpa_path, project_paths):
    """Create interactive matplotlib visualization of SubModelParts."""
    parsed_mdpa = parse_mdpa_data(mdpa_path)
    nodes = parsed_mdpa.nodes
    submodel_nodes = parsed_mdpa.submodel_nodes
    connections = parsed_mdpa.connections
    entity_submodels = parsed_mdpa.entity_submodels
    entity_records = parsed_mdpa.entity_records
    mdpa_dir = os.path.dirname(os.path.abspath(mdpa_path))
    mdpa_stem = os.path.splitext(os.path.basename(mdpa_path))[0]

    def _discover_project_json_files(search_dir):
        """Collect likely stage/project JSON files from a directory."""
        discovered = []
        seen = set()
        preferred_names = [
            f"{mdpa_stem}_stages_input.json",
            f"{mdpa_stem}_stages.json",
            "ProjectParameters.json",
        ]

        for name in preferred_names:
            candidate = os.path.join(search_dir, name)
            if os.path.isfile(candidate):
                norm = os.path.normcase(os.path.normpath(candidate))
                if norm not in seen:
                    seen.add(norm)
                    discovered.append(candidate)

        # Include additional stage-like JSON files as fallback.
        try:
            for name in sorted(os.listdir(search_dir)):
                low = name.lower()
                if not low.endswith('.json'):
                    continue
                if 'stage' not in low and low != 'projectparameters.json':
                    continue
                candidate = os.path.join(search_dir, name)
                if not os.path.isfile(candidate):
                    continue
                norm = os.path.normcase(os.path.normpath(candidate))
                if norm in seen:
                    continue
                seen.add(norm)
                discovered.append(candidate)
        except Exception:
            pass

        return discovered

    def _resolve_input_path(path):
        """Resolve input path against mdpa directory first, then cwd."""
        raw = (path or '').strip()
        expanded = os.path.expanduser(raw.strip("\"'"))

        if os.path.isabs(expanded):
            return expanded

        candidate = os.path.normpath(os.path.join(mdpa_dir, expanded))
        if os.path.exists(candidate):
            return candidate

        return expanded

    expanded_project_paths = []
    expanded_seen = set()
    for p in project_paths:
        resolved = _resolve_input_path(p)
        if os.path.isdir(resolved):
            for candidate in _discover_project_json_files(resolved):
                norm = os.path.normcase(os.path.normpath(candidate))
                if norm not in expanded_seen:
                    expanded_seen.add(norm)
                    expanded_project_paths.append(candidate)
        elif os.path.isfile(resolved):
            norm = os.path.normcase(os.path.normpath(resolved))
            if norm not in expanded_seen:
                expanded_seen.add(norm)
                expanded_project_paths.append(resolved)
        elif p:
            print(f"Warning: Could not find project path '{p}'")

    if expanded_project_paths and project_paths:
        print(f"Expanded --project input to: {', '.join(expanded_project_paths)}")

    project_paths = expanded_project_paths

    # Auto-detect common stage/project files when -p/--project is not provided.
    # This keeps stage buttons usable for typical folder layouts.
    if not project_paths:
        project_paths = _discover_project_json_files(mdpa_dir)
        if project_paths:
            print(f"Auto-detected project file(s): {', '.join(project_paths)}")

    if project_paths:
        processes_map, stage_names, stage_referenced_submodels = parse_project_parameters(project_paths, base_dir=mdpa_dir)

        # Resilient fallback: if stage-tagged process extraction misses stage names,
        # read them directly from top-level "stages" blocks.
        if not stage_names:
            recovered_stage_names = []
            seen_recovered = set()

            def _remember_stage_name(name):
                if isinstance(name, str) and name and name not in seen_recovered:
                    seen_recovered.add(name)
                    recovered_stage_names.append(name)

            def _resolve_local_json(path):
                raw = (path or '').strip()
                expanded = os.path.expanduser(raw.strip("\"'"))
                if os.path.isabs(expanded):
                    return expanded if os.path.isfile(expanded) else None
                candidate = os.path.normpath(os.path.join(mdpa_dir, expanded))
                if os.path.isfile(candidate):
                    return candidate
                return expanded if os.path.isfile(expanded) else None

            for path in project_paths:
                resolved = _resolve_local_json(path)
                if not resolved:
                    continue
                try:
                    with open(resolved, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue

                stages = data.get('stages') if isinstance(data, dict) else None
                if isinstance(stages, dict):
                    for key in stages.keys():
                        _remember_stage_name(key)
                elif isinstance(stages, list):
                    for idx, item in enumerate(stages, start=1):
                        if isinstance(item, dict):
                            _remember_stage_name(
                                item.get('name')
                                or item.get('stage_name')
                                or item.get('id')
                                or f"stage_{idx}"
                            )
                        else:
                            _remember_stage_name(f"stage_{idx}")

            if recovered_stage_names:
                stage_names = recovered_stage_names
                print(
                    f"Recovered stage names from JSON 'stages' blocks: {', '.join(stage_names)}"
                )
    else:
        processes_map, stage_names, stage_referenced_submodels = {}, [], {}
    
    print(f"\n{'='*70}")
    print(f"MDPA Visualization")
    print(f"{'='*70}")
    print(f"Loaded: {len(nodes)} nodes from {mdpa_path}")
    print(f"Found: {len(submodel_nodes)} SubModelParts")
    if processes_map:
        total_procs = sum(len(set(p)) for p in processes_map.values())
        print(f"Loaded: {total_procs} unique processes from {', '.join(project_paths)}")
    if stage_names:
        print(f"Discovered stages: {', '.join(stage_names)}")
    else:
        print("Discovered stages: none")
        if not project_paths:
            print("Tip: pass -p/--project with a stage JSON file to enable per-stage buttons.")
    print(f"{'='*70}\n")
    print("KEYBOARD SHORTCUTS:")
    print("  - Press 1-9, 0: Show processes for SubModelPart #1-10")
    print("  - Click on nodes: See which SubModelParts they belong to")
    print("  - Use Stage list in right panel to show active SubModelParts for a selected stage")
    print("  - Use right panel Find/Show/Hide/Invert to filter visibility")
    print("  - Press z: Progressive zoom to the last clicked node")
    print("  - Press r: Reset zoom to full extent")
    print(f"{'='*70}\n")
    
    if not nodes:
        print("No nodes found in mdpa file!")
        return

    # Prepare data
    node_ids = sorted(nodes.keys())
    node_coords = np.array([nodes[nid] for nid in node_ids])
    
    # Assign color per SubModelPart
    subnames = sorted(submodel_nodes.keys())
    subname_to_idx = {name: idx for idx, name in enumerate(subnames)}

    def _color_distance(c1, c2):
        # Weighted RGB distance with luminance emphasis.
        dr = c1[0] - c2[0]
        dg = c1[1] - c2[1]
        db = c1[2] - c2[2]
        lum1 = 0.2126 * c1[0] + 0.7152 * c1[1] + 0.0722 * c1[2]
        lum2 = 0.2126 * c2[0] + 0.7152 * c2[1] + 0.0722 * c2[2]
        dl = lum1 - lum2
        return (dr * dr + dg * dg + db * db) + 0.5 * (dl * dl)

    def _build_candidate_colors(count):
        # Build a large candidate pool and pick from it greedily.
        candidates = []
        golden = 0.618033988749895
        h = 0.0
        while len(candidates) < count:
            h = (h + golden) % 1.0
            for s, v in ((0.92, 0.90), (0.78, 0.95), (0.88, 0.78), (0.70, 0.88)):
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                candidates.append((r, g, b, 1.0))
                if len(candidates) >= count:
                    break
        return candidates

    # Build co-occurrence graph from nodes that belong to multiple submodel parts.
    cooccur = {name: defaultdict(int) for name in subnames}
    node_to_subs_for_graph = {nid: [] for nid in nodes.keys()}
    for sname in subnames:
        for nid in submodel_nodes.get(sname, set()):
            node_to_subs_for_graph[nid].append(sname)
    for memberships in node_to_subs_for_graph.values():
        if len(memberships) < 2:
            continue
        uniq = sorted(set(memberships))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a = uniq[i]
                b = uniq[j]
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    # Co-occurrence-aware greedy color assignment:
    # submodels that appear together are forced apart in color space.
    candidates = _build_candidate_colors(max(4 * len(subnames), 32))
    degrees = {
        name: sum(cooccur[name].values())
        for name in subnames
    }
    order = sorted(subnames, key=lambda n: (degrees[n], len(submodel_nodes.get(n, set()))), reverse=True)

    assigned = {}
    used = set()
    for name in order:
        best_idx = None
        best_score = -1.0
        for idx, cand in enumerate(candidates):
            if idx in used:
                continue
            score = 0.0
            min_dist = 1e9
            for other, other_color in assigned.items():
                w = cooccur[name].get(other, 0)
                d = _color_distance(cand, other_color)
                # Heavily weight distances to co-occurring neighbors.
                score += (1.0 + 3.0 * w) * d
                min_dist = min(min_dist, d)
            # Encourage global separation too.
            if assigned:
                score += 0.8 * min_dist
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            best_idx = 0
        assigned[name] = candidates[best_idx]
        used.add(best_idx)

    def _color_for_submodel(name):
        return assigned.get(name, (0.2, 0.2, 0.2, 1.0))

    subname_to_hue = {}
    for name in subnames:
        r, g, b, _a = _color_for_submodel(name)
        h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
        subname_to_hue[name] = h

    def _high_contrast_order(names):
        """Order names so adjacent segments have high hue contrast."""
        ordered = sorted(names, key=lambda n: subname_to_hue.get(n, 0.0))
        result = []
        left = 0
        right = len(ordered) - 1
        pick_right = True
        while left <= right:
            if pick_right:
                result.append(ordered[right])
                right -= 1
            else:
                result.append(ordered[left])
                left += 1
            pick_right = not pick_right
        return result

    def _edge_color_for_segment(fill_rgba, seg_idx):
        """Build a high-contrast edge color per wedge segment."""
        r, g, b = fill_rgba[:3]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # Rotate hue and alternate value for stronger neighboring contrast.
        eh = (h + 0.47 + 0.11 * (seg_idx % 4)) % 1.0
        es = min(1.0, max(0.70, s + 0.15))
        ev = 0.15 if seg_idx % 2 == 0 else 0.95
        er, eg, eb = colorsys.hsv_to_rgb(eh, es, ev)
        return (er, eg, eb, 1.0)

    # Build node -> submodelpart membership mapping (preserve consistent ordering)
    node_to_subs = {nid: [] for nid in node_ids}
    for sname in subnames:
        for nid in sorted(submodel_nodes.get(sname, [])):
            node_to_subs[nid].append(sname)

    # Build XY -> node ids mapping to report coincident nodes (same XY location).
    xy_to_node_ids = defaultdict(list)
    node_xy = {}
    for nid in node_ids:
        x, y, _ = nodes[nid]
        key = (x, y)
        node_xy[nid] = key
        xy_to_node_ids[key].append(nid)

    # Group coincident nodes by XY for depth-based stacked rendering.
    xy_groups = {}
    for key, ids in xy_to_node_ids.items():
        group = {
            'node_ids': sorted(ids),
            'submodels': [],
        }
        seen_subs = set()
        for nid in group['node_ids']:
            for sname in node_to_subs.get(nid, []):
                if sname in seen_subs:
                    continue
                seen_subs.add(sname)
                group['submodels'].append(sname)
        xy_groups[key] = group

    # Create plot (use first 2D projection for simplicity)
    fig, ax = plt.subplots(figsize=(14, 11))
    # Reserve a stable right-side margin for UI controls so they stay outside
    # the data area and never overlap plotted points.
    fig.subplots_adjust(left=0.07, right=0.70, bottom=0.08, top=0.95)

    # Set limits explicitly from node cloud so custom patches render in full domain.
    x_min, x_max = float(np.min(node_coords[:, 0])), float(np.max(node_coords[:, 0]))
    y_min, y_max = float(np.min(node_coords[:, 1])), float(np.max(node_coords[:, 1]))
    dx = max(x_max - x_min, 1.0)
    dy = max(y_max - y_min, 1.0)
    pad_x = 0.03 * dx
    pad_y = 0.03 * dy
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    # We'll draw nodes via a helper so we can keep marker sizes constant in screen
    # space. Store patches we create so they can be removed and redrawn on zoom/pan.
    node_patches = []
    hidden_submodels = set()

    # Desired marker size in points (matplotlib points; 1 point = 1/72 inch)
    marker_point_size = 6.0
    selected_stage = 'All'
    stage_inactive_submodels = set()
    mesh_visible = False
    mesh_artists = []
    normals_visible = False
    normal_artists = []
    normal_length_scale = 1.0
    equal_aspect = False

    def _get_processes_for_submodel(submodel_name):
        """Lookup process labels by exact or normalized SubModelPart name."""
        exact = processes_map.get(submodel_name)
        if exact is not None:
            return exact
        return processes_map.get(_normalize_submodelpart_name(submodel_name), [])

    def _rebuild_stage_activity():
        stage_inactive_submodels.clear()
        if selected_stage == 'All':
            return

        # Primary source: all SubModelParts referenced by selected stage
        # (processes, operations/modelers, and stage domain lists).
        referenced_for_stage = stage_referenced_submodels.get(selected_stage, set())
        if referenced_for_stage:
            for name in subnames:
                if _normalize_submodelpart_name(name) not in referenced_for_stage:
                    stage_inactive_submodels.add(name)
            return

        # Fallback when no stage-tagged references were collected.
        for name in subnames:
            procs = _get_processes_for_submodel(name)
            if not any(_stage_from_process_label(p) == selected_stage for p in procs):
                stage_inactive_submodels.add(name)

    def _is_active_for_selected_stage(submodel_name):
        return submodel_name not in stage_inactive_submodels

    def _make_wedge_path(theta2, theta1, nsteps=12):
        # Create a wedge (sector) path in marker coordinate space (radius=1)
        # theta2, theta1 in degrees (start, end)
        th = np.linspace(np.deg2rad(theta2), np.deg2rad(theta1), nsteps)
        verts = [(0.0, 0.0)]
        verts += [(np.cos(t), np.sin(t)) for t in th]
        verts.append((0.0, 0.0))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
        return Path(verts, codes)


    def draw_nodes():
        # Remove previous patches
        for p in node_patches:
            try:
                p.remove()
            except Exception:
                pass
        node_patches.clear()

        # Compute pixel radius from point size (size for scatter is area in points^2)
        size_area = (marker_point_size) ** 2

        # Batch draw one marker per XY coordinate so coincident nodes become a
        # concentric depth stack instead of a segmented blob.
        single_groups = defaultdict(list)
        segmented_groups = defaultdict(list)

        def _add_segmented_node(x, y, subs, size_scale, alpha, zorder):
            subs_sorted = _high_contrast_order(subs)
            nseg = len(subs_sorted)
            for i, sname in enumerate(subs_sorted):
                segmented_groups[(nseg, i, sname, size_scale, alpha, zorder)].append((x, y))

        for key, group in xy_groups.items():
            x, y = key
            node_ids_in_group = sorted(group['node_ids'])

            if len(node_ids_in_group) == 1:
                # Single node at this coordinate: keep the existing simple circle.
                nid = node_ids_in_group[0]
                subs = [s for s in node_to_subs.get(nid, []) if s not in hidden_submodels]
                if not subs:
                    continue
                if len(subs) == 1:
                    single_groups[(subs[0], 1.0, 0.9, 2.0)].append((x, y))
                else:
                    _add_segmented_node(x, y, subs, 1.0, 0.95, 3.0)
            else:
                # Multiple nodes at the same XY: create depth layers where each
                # deeper layer is slightly larger than the one above it.
                depth_count = len(node_ids_in_group)
                # Draw deeper layers first so they remain underneath smaller, upper layers.
                for depth_idx, nid in enumerate(reversed(node_ids_in_group)):
                    stack_level = depth_count - depth_idx - 1
                    size_scale = 1.0 + 0.55 * stack_level
                    alpha = max(0.30, 0.82 - 0.12 * stack_level)
                    # Deeper/larger layers must stay below upper/smaller layers.
                    zorder = 3 - stack_level * 0.01

                    subs = [s for s in node_to_subs.get(nid, []) if s not in hidden_submodels]
                    if not subs:
                        continue
                    if len(subs) == 1:
                        single_groups[(subs[0], size_scale, alpha, zorder)].append((x, y))
                    else:
                        _add_segmented_node(x, y, subs, size_scale, alpha, zorder)

        # Clamp marker size to avoid extreme sizes when a coordinate is heavily overlapped.
        max_size = max(1000, size_area * 100)
        size_area_clamped = min(size_area, max_size)

        for (sname, size_scale, alpha, zorder), coords in single_groups.items():
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            color = _color_for_submodel(sname)
            sc = ax.scatter(
                xs,
                ys,
                s=size_area_clamped * (size_scale ** 2),
                marker='o',
                facecolors=[color],
                edgecolors='black',
                linewidths=0.55,
                alpha=alpha,
                zorder=zorder
            )
            node_patches.append(sc)

        for (nseg, i, sname, size_scale, alpha, zorder), coords in segmented_groups.items():
            start_angle = 90.0
            theta1 = start_angle - (i) * (360.0 / nseg)
            theta2 = start_angle - (i + 1) * (360.0 / nseg)
            path = _make_wedge_path(theta2, theta1)
            color = _color_for_submodel(sname)
            edge = _edge_color_for_segment(color, i)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            sc = ax.scatter(
                xs,
                ys,
                s=size_area_clamped * (size_scale ** 2),
                marker=path,
                facecolors=[color],
                edgecolors=[edge],
                linewidths=[0.65],
                alpha=alpha,
                zorder=zorder + 0.002
            )
            node_patches.append(sc)

        # NOTE: do not call draw here; callers trigger canvas refresh as needed.
        return

    def draw_mesh():
        """Draw or remove connectivity lines for currently active nodes."""
        for artist in mesh_artists:
            artist.remove()
        mesh_artists.clear()
        if not mesh_visible:
            return

        active_node_ids = {
            nid for nid, memberships in node_to_subs.items()
            if any(name not in hidden_submodels for name in memberships)
        }
        for entity_type, node_ids_in_connection in connections:
            for path in _connection_paths(entity_type, node_ids_in_connection):
                if len(path) < 2:
                    continue
                if not all(nid in nodes and nid in active_node_ids for nid in path):
                    continue
                xs = [nodes[nid][0] for nid in path]
                ys = [nodes[nid][1] for nid in path]
                (line,) = ax.plot(xs, ys, color='black', linewidth=0.8, zorder=1)
                mesh_artists.append(line)

    def draw_normals():
        """Draw or remove normals for active line entities by SubModelPart."""
        for artist in normal_artists:
            artist.remove()
        normal_artists.clear()
        if not normals_visible:
            return

        normal_length = 0.06 * max(dx, dy) * normal_length_scale
        for block_type, entity_type, entity_id, node_ids_in_entity in entity_records:
            memberships = entity_submodels.get((block_type, entity_id), set())
            active_memberships = [
                name for name in memberships if name not in hidden_submodels
            ]
            if not active_memberships:
                continue
            type_name = (entity_type or '').lower()
            is_interface = '2plus2n' in type_name or '3plus3n' in type_name
            segments = _normal_segments(entity_type, node_ids_in_entity)
            if block_type != 'Conditions' and not is_interface and not (
                'line' in type_name or len(node_ids_in_entity) in (2, 3)
            ):
                continue
            for submodel_name in active_memberships:
                for segment in segments:
                    start = nodes.get(segment[0])
                    end = nodes.get(segment[-1])
                    if start is None or end is None:
                        continue
                    normal = _normal_vector(start, end, normal_length)
                    if normal is None:
                        continue
                    origin = np.array([(start[0] + end[0]) / 2,
                                       (start[1] + end[1]) / 2])
                    vector = ax.quiver(
                        origin[0], origin[1], normal[0], normal[1],
                        color=_color_for_submodel(submodel_name),
                        angles='xy', scale_units='xy', scale=1,
                        width=0.003, zorder=2,
                    )
                    normal_artists.append(vector)

    # Initial draw
    draw_nodes()
    # Request a single canvas update now that patches are created
    fig.canvas.draw_idle()

    # Add top-bar controls above the Find box.
    save_plot_ax = fig.add_axes([0.86, 0.885, 0.13, 0.03])
    save_plot_btn = Button(save_plot_ax, 'Save plot', hovercolor='0.975')

    def _save_plot(_event):
        fig.canvas.draw()
        plot_bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.dpi_scale_trans.inverted()
        ).padded(0.1)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(mdpa_dir, f'mdpa_plot_{timestamp}.png')
        fig.savefig(output_path, bbox_inches=plot_bbox, dpi=fig.dpi)
        print(f'Plot saved to {output_path}')

    save_plot_btn.on_clicked(_save_plot)

    mesh_btn_ax = fig.add_axes([0.72, 0.955, 0.085, 0.03])
    mesh_btn = Button(mesh_btn_ax, 'Draw mesh', hovercolor='0.975')
    normals_btn_ax = fig.add_axes([0.81, 0.955, 0.085, 0.03])
    normals_btn = Button(normals_btn_ax, 'Draw normals', hovercolor='0.975')
    decrease_btn_ax = fig.add_axes([0.90, 0.955, 0.09, 0.03])
    increase_btn_ax = fig.add_axes([0.90, 0.92, 0.09, 0.03])
    decrease_btn = Button(decrease_btn_ax, 'point size -', hovercolor='0.975')
    increase_btn = Button(increase_btn_ax, 'point size +', hovercolor='0.975')

    def _toggle_mesh(_event):
        nonlocal mesh_visible
        mesh_visible = not mesh_visible
        draw_mesh()
        fig.canvas.draw_idle()

    mesh_btn.on_clicked(_toggle_mesh)

    def _toggle_normals(_event):
        nonlocal normals_visible
        normals_visible = not normals_visible
        draw_normals()
        fig.canvas.draw_idle()

    normals_btn.on_clicked(_toggle_normals)

    aspect_btn_ax = fig.add_axes([0.72, 0.885, 0.13, 0.03])
    aspect_btn = Button(aspect_btn_ax, 'Aspect: free', hovercolor='0.975')

    def _toggle_aspect(_event):
        nonlocal equal_aspect
        equal_aspect = not equal_aspect
        ax.set_aspect('equal' if equal_aspect else 'auto', adjustable='box')
        aspect_btn.label.set_text('Aspect: equal' if equal_aspect else 'Aspect: free')
        fig.canvas.draw_idle()

    aspect_btn.on_clicked(_toggle_aspect)

    normal_decrease_ax = fig.add_axes([0.72, 0.92, 0.085, 0.03])
    normal_increase_ax = fig.add_axes([0.81, 0.92, 0.085, 0.03])
    normal_decrease_btn = Button(normal_decrease_ax, 'Normal -', hovercolor='0.975')
    normal_increase_btn = Button(normal_increase_ax, 'Normal +', hovercolor='0.975')

    def _change_normal_scale(factor):
        nonlocal normal_length_scale
        normal_length_scale = normal_length_scale * factor
        draw_normals()
        fig.canvas.draw_idle()

    normal_decrease_btn.on_clicked(lambda _event: _change_normal_scale(1.0 / 1.1))
    normal_increase_btn.on_clicked(lambda _event: _change_normal_scale(1.1))

    def _clamp_marker_size(val):
        return max(1.0, min(val, 200.0))

    def _change_marker_size(delta):
        nonlocal marker_point_size
        marker_point_size = _clamp_marker_size(marker_point_size + delta)
        print(f"Marker size set to {marker_point_size} pt")
        draw_nodes()
        draw_mesh()
        draw_normals()
        fig.canvas.draw_idle()

    decrease_btn.on_clicked(lambda _event: _change_marker_size(-1.0))
    increase_btn.on_clicked(lambda _event: _change_marker_size(1.0))

    # Panel controls: search + bulk visibility operations + checkbox list.
    search_ax = fig.add_axes([0.74, 0.84, 0.25, 0.03])
    search_box = TextBox(search_ax, 'Find ', initial='')

    btn_show_ax = fig.add_axes([0.72, 0.80, 0.085, 0.03])
    btn_hide_ax = fig.add_axes([0.81, 0.80, 0.085, 0.03])
    btn_inv_ax = fig.add_axes([0.90, 0.80, 0.09, 0.03])
    show_btn = Button(btn_show_ax, 'Show', hovercolor='0.95')
    hide_btn = Button(btn_hide_ax, 'Hide', hovercolor='0.95')
    invert_btn = Button(btn_inv_ax, 'Invert', hovercolor='0.95')

    stage_options = ['All'] + stage_names
    # Keep a fixed Stage panel height and provide a scrollbar so all stages are reachable.
    stage_height = 0.24
    stage_top = 0.78
    stage_bottom = stage_top - stage_height
    stage_ax = fig.add_axes([0.72, stage_bottom, 0.252, stage_height])
    stage_scroll_ax = fig.add_axes([0.975, stage_bottom, 0.015, stage_height])
    stage_ax.set_title('Stage', fontsize=8)

    stage_selector = None
    suppress_stage_callback = {'value': False}
    suppress_stage_scroll_callback = {'value': False}
    stage_offset = {'value': 0}
    stage_per_page = max(6, int((stage_height - 0.03) / 0.024))
    max_stage_offset = max(0, len(stage_options) - stage_per_page)

    slider_max = max(1, max_stage_offset)
    stage_scroll = Slider(
        ax=stage_scroll_ax,
        label='',
        valmin=0,
        valmax=slider_max,
        valinit=slider_max,
        valstep=1,
        orientation='vertical'
    )
    stage_scroll_ax.set_facecolor('#f2f2f2')
    if hasattr(stage_scroll, 'valtext'):
        stage_scroll.valtext.set_visible(False)

    def _current_stage_page_options():
        start = stage_offset['value']
        end = min(len(stage_options), start + stage_per_page)
        return stage_options[start:end]

    def _render_stage_widget():
        nonlocal stage_selector
        stage_ax.cla()
        stage_ax.set_title('Stage', fontsize=8)
        page_options = _current_stage_page_options()
        page_states = [opt == selected_stage for opt in page_options]
        stage_selector = CheckButtons(stage_ax, page_options, page_states)
        for lbl in stage_selector.labels:
            lbl.set_fontsize(8)
        stage_selector.on_clicked(_on_stage_selected)

    def _on_stage_scroll(value):
        if suppress_stage_scroll_callback['value']:
            return
        stage_offset['value'] = max(
            0, min(max_stage_offset, int(round(slider_max - value)))
        )
        _render_stage_widget()
        fig.canvas.draw_idle()

    def _on_stage_mouse_wheel(event):
        if event.inaxes not in (stage_ax, stage_scroll_ax):
            return
        if max_stage_offset <= 0:
            return

        if event.button == 'up':
            new_offset = max(0, stage_offset['value'] - 1)
        elif event.button == 'down':
            new_offset = min(max_stage_offset, stage_offset['value'] + 1)
        else:
            return

        if new_offset != stage_offset['value']:
            stage_offset['value'] = new_offset
            suppress_stage_scroll_callback['value'] = True
            stage_scroll.set_val(slider_max - new_offset)
            suppress_stage_scroll_callback['value'] = False
            _render_stage_widget()
            fig.canvas.draw_idle()

    checks_bottom = 0.12
    checks_top = stage_bottom - 0.02
    checks_height = max(0.25, checks_top - checks_bottom)
    checks_ax = fig.add_axes([0.72, checks_bottom, 0.252, checks_height])
    checks_scroll_ax = fig.add_axes([0.975, checks_bottom, 0.015, checks_height])
    checks_ax.set_title('Visible SubModelParts', fontsize=8)
    checks_scroll_ax.set_title('', fontsize=8)

    checkbox_state = {name: True for name in subnames}
    filter_query = ''
    filter_query_lower = ''
    suppress_checkbox_callback = {'value': False}
    suppress_scroll_callback = {'value': False}
    subnames_lower = {name: name.lower() for name in subnames}

    # Paged checkbox list + slider-based vertical scroller.
    checks_per_page = max(8, int((checks_height - 0.04) / 0.022))
    checks_offset = {'value': 0}
    max_scroll_offset = max(0, len(subnames) - checks_per_page)
    visibility_checks = None
    rectangles = None
    lines = None

    checks_slider_max = max(1, max_scroll_offset)
    checks_scroll = Slider(
        ax=checks_scroll_ax,
        label='',
        valmin=0,
        valmax=checks_slider_max,
        valinit=checks_slider_max,
        valstep=1,
        orientation='vertical'
    )
    checks_scroll_ax.set_facecolor('#f2f2f2')
    if hasattr(checks_scroll, 'valtext'):
        checks_scroll.valtext.set_visible(False)

    def _current_page_names():
        start = checks_offset['value']
        end = min(len(subnames), start + checks_per_page)
        return subnames[start:end]

    def _matches_filter(name):
        if not filter_query_lower:
            return True
        return filter_query_lower in subnames_lower[name]

    def _set_checkbox_visuals():
        if visibility_checks is None:
            return
        for i, lbl in enumerate(visibility_checks.labels):
            name = lbl.get_text()
            color = _color_for_submodel(name)
            match = _matches_filter(name)
            is_active = checkbox_state[name]

            lbl.set_fontsize(7)
            lbl.set_color(color)
            lbl.set_alpha(1.0)
            # Highlight Find matches among currently visible stage entries.
            lbl.set_fontweight('bold' if (filter_query and match) else 'normal')
            lbl.set_visible(True)

            if rectangles is not None and i < len(rectangles):
                rect = rectangles[i]
                rect.set_visible(True)
                rect.set_facecolor(color)
                rect.set_alpha(0.25)
                rect.set_edgecolor(color)

            if lines is not None and i < len(lines):
                pair = lines[i]
                for line in pair:
                    line.set_visible(True)
                    line.set_color(color)
                    line.set_linewidth(1.5)

    def _sync_hidden_from_checkbox_state():
        hidden_submodels.clear()
        for name, is_active in checkbox_state.items():
            if not is_active:
                hidden_submodels.add(name)

    def _set_checkbox_state(name, visible):
        if checkbox_state[name] == visible:
            return
        checkbox_state[name] = visible

        # If the entry is on current page, keep the widget in sync.
        if visibility_checks is not None:
            status = visibility_checks.get_status()
            for i, lbl in enumerate(visibility_checks.labels):
                if lbl.get_text() != name:
                    continue
                if status[i] != visible:
                    suppress_checkbox_callback['value'] = True
                    visibility_checks.set_active(i)
                    suppress_checkbox_callback['value'] = False
                break

    def _render_checks_widget():
        nonlocal visibility_checks, rectangles, lines
        checks_ax.cla()
        checks_ax.set_title('Visible SubModelParts', fontsize=8)

        page_names = _current_page_names()
        page_states = [checkbox_state[n] for n in page_names]
        visibility_checks = CheckButtons(checks_ax, page_names, page_states)
        visibility_checks.on_clicked(_toggle_submodel_visibility)

        rectangles = getattr(visibility_checks, 'rectangles', None)
        lines = getattr(visibility_checks, 'lines', None)
        _set_checkbox_visuals()

    def _toggle_submodel_visibility(label):
        if suppress_checkbox_callback['value']:
            return
        checkbox_state[label] = not checkbox_state[label]
        _sync_hidden_from_checkbox_state()
        _set_checkbox_visuals()
        draw_nodes()
        draw_mesh()
        draw_normals()
        fig.canvas.draw_idle()

    def _apply_bulk(action):
        changed = False
        affected = 0
        for name in subnames:
            # Invert acts on the current checkbox choices globally.
            # Show/Hide operate only on submodels active for selected stage.
            if action != 'invert' and not _is_active_for_selected_stage(name):
                continue
            # Invert is global by design: swap visible/hidden for all entries.
            # Show/Hide apply to current Find matches.
            if action != 'invert' and not _matches_filter(name):
                continue
            affected += 1
            if action == 'show':
                target = True
            elif action == 'hide':
                target = False
            else:  # invert
                target = not checkbox_state[name]
            if checkbox_state[name] != target:
                _set_checkbox_state(name, target)
                changed = True
        if changed:
            _sync_hidden_from_checkbox_state()
            _render_checks_widget()
            draw_nodes()
            draw_mesh()
            draw_normals()
            fig.canvas.draw_idle()
        if action == 'invert':
            print(f"Bulk action 'invert' applied globally to {affected} SubModelParts.")
        elif filter_query:
            print(f"Bulk action '{action}' applied to {affected} filtered SubModelParts.")

    def _commit_search_query(text):
        nonlocal filter_query, filter_query_lower
        new_query = (text or '').strip()
        new_query_lower = new_query.lower()
        if new_query == filter_query and new_query_lower == filter_query_lower:
            return
        filter_query = new_query
        filter_query_lower = new_query_lower
        _render_checks_widget()
        fig.canvas.draw_idle()

    def _on_checks_scroll(value):
        if suppress_scroll_callback['value']:
            return
        checks_offset['value'] = max(
            0, min(max_scroll_offset, int(round(checks_slider_max - value)))
        )
        _render_checks_widget()
        fig.canvas.draw_idle()

    def _on_checks_mouse_wheel(event):
        if event.inaxes != checks_ax:
            return
        if event.button == 'up':
            new_offset = max(0, checks_offset['value'] - 1)
        elif event.button == 'down':
            new_offset = min(max_scroll_offset, checks_offset['value'] + 1)
        else:
            return

        if new_offset != checks_offset['value']:
            checks_offset['value'] = new_offset
            suppress_scroll_callback['value'] = True
            checks_scroll.set_val(checks_slider_max - new_offset)
            suppress_scroll_callback['value'] = False
            _render_checks_widget()
            fig.canvas.draw_idle()

    def _on_plot_mouse_wheel(event):
        """Zoom on plot with touchpad/mouse wheel using cursor-centered scaling."""
        if event.inaxes != ax:
            return

        if event.button == 'up':
            scale = 1.0 / 1.15  # zoom in
        elif event.button == 'down':
            scale = 1.15  # zoom out
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_width = cur_xlim[1] - cur_xlim[0]
        cur_height = cur_ylim[1] - cur_ylim[0]

        x_center = event.xdata if event.xdata is not None else 0.5 * (cur_xlim[0] + cur_xlim[1])
        y_center = event.ydata if event.ydata is not None else 0.5 * (cur_ylim[0] + cur_ylim[1])

        # Keep cursor position stable in view while scaling limits.
        rel_x = 0.0 if cur_width == 0.0 else (x_center - cur_xlim[0]) / cur_width
        rel_y = 0.0 if cur_height == 0.0 else (y_center - cur_ylim[0]) / cur_height

        new_width = cur_width * scale
        new_height = cur_height * scale

        min_width = max(dx * 1.0e-6, 1.0e-12)
        min_height = max(dy * 1.0e-6, 1.0e-12)
        new_width = max(new_width, min_width)
        new_height = max(new_height, min_height)

        x_min = x_center - rel_x * new_width
        x_max = x_center + (1.0 - rel_x) * new_width
        y_min = y_center - rel_y * new_height
        y_max = y_center + (1.0 - rel_y) * new_height

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Wheel zoom resets progressive keyboard zoom chain to avoid mixed states.
        zoom_state['node_idx'] = None
        zoom_state['half_width'] = None
        zoom_state['half_height'] = None
        zoom_state['level'] = 0

        fig.canvas.draw_idle()

    # Debounce live search updates to avoid expensive full checkbox redraw per keystroke.
    search_debounce = {'query': ''}
    search_timer = fig.canvas.new_timer(interval=120)

    def _flush_debounced_search():
        _commit_search_query(search_debounce['query'])

    search_timer.add_callback(_flush_debounced_search)

    def _on_search_changed(text):
        search_debounce['query'] = text or ''
        try:
            search_timer.stop()
            search_timer.start()
        except Exception:
            # Fallback for backends without timer support.
            _commit_search_query(text)

    def _on_search_submit(text):
        search_debounce['query'] = text or ''
        try:
            search_timer.stop()
        except Exception:
            pass
        _commit_search_query(text)

    def _on_stage_selected(label):
        nonlocal selected_stage
        if suppress_stage_callback['value']:
            return

        # Enforce single-selection behavior for stage list.
        if label == selected_stage:
            # CheckButtons toggles clicked item; restore selected state.
            _render_stage_widget()
            fig.canvas.draw_idle()
            return

        selected_stage = label
        _rebuild_stage_activity()

        # Apply selected stage using the same mechanism as Hide/Show:
        # active submodels => checked (visible), inactive => unchecked (hidden).
        changed = False
        for name in subnames:
            target = _is_active_for_selected_stage(name)
            if checkbox_state[name] != target:
                _set_checkbox_state(name, target)
                changed = True
        # Always re-sync hidden-state from checkboxes after stage application
        # to avoid stale visibility when no toggle event is emitted.
        _sync_hidden_from_checkbox_state()

        _render_stage_widget()
        _render_checks_widget()
        draw_nodes()
        draw_mesh()
        draw_normals()
        fig.canvas.draw_idle()
        active_count = sum(1 for n in subnames if _is_active_for_selected_stage(n))
        print(f"Selected stage: {selected_stage} ({active_count} active SubModelParts)")

    # Apply filter live while typing and also on Enter.
    search_box.on_text_change(_on_search_changed)
    search_box.on_submit(_on_search_submit)
    show_btn.on_clicked(lambda _e: _apply_bulk('show'))
    hide_btn.on_clicked(lambda _e: _apply_bulk('hide'))
    invert_btn.on_clicked(lambda _e: _apply_bulk('invert'))
    stage_scroll.on_changed(_on_stage_scroll)
    checks_scroll.on_changed(_on_checks_scroll)
    fig.canvas.mpl_connect('scroll_event', _on_stage_mouse_wheel)
    fig.canvas.mpl_connect('scroll_event', _on_checks_mouse_wheel)
    fig.canvas.mpl_connect('scroll_event', _on_plot_mouse_wheel)

    _rebuild_stage_activity()
    _render_stage_widget()
    _render_checks_widget()
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'MDPA Visualization: {len(subnames)} SubModelParts, {len(nodes)} nodes\nPress number keys (1-9, 0) to see processes')
    ax.grid(True, alpha=0.3)
    
    # Add click interactivity for nodes
    # Store original view limits for reset
    orig_xlim = ax.get_xlim()
    orig_ylim = ax.get_ylim()
    last_selected = {'idx': None}
    zoom_state = {'node_idx': None, 'half_width': None, 'half_height': None, 'level': 0}

    def _zoom_to_region(cx, cy, half_width, half_height):
        ax.set_xlim(cx - half_width, cx + half_width)
        ax.set_ylim(cy - half_height, cy + half_height)
        fig.canvas.draw_idle()

    def _zoom_to_node_index(node_idx, progressive=False):
        cx, cy = node_coords[node_idx, 0], node_coords[node_idx, 1]

        if progressive:
            # Progressive sequence: first zoom is moderate, each next one zooms further.
            min_half_w = max((marker_point_size / 72.0) * dx * 0.12, dx * 1.0e-6)
            min_half_h = max((marker_point_size / 72.0) * dy * 0.12, dy * 1.0e-6)

            if zoom_state['node_idx'] != node_idx or zoom_state['half_width'] is None or zoom_state['half_height'] is None:
                half_w = 0.18 * (orig_xlim[1] - orig_xlim[0])
                half_h = 0.18 * (orig_ylim[1] - orig_ylim[0])
                level = 1
            else:
                half_w = max(zoom_state['half_width'] * 0.58, min_half_w)
                half_h = max(zoom_state['half_height'] * 0.58, min_half_h)
                level = zoom_state['level'] + 1

            zoom_state['node_idx'] = node_idx
            zoom_state['half_width'] = half_w
            zoom_state['half_height'] = half_h
            zoom_state['level'] = level
            _zoom_to_region(cx, cy, half_w, half_h)
        else:
            # Single-step zoom used by double-click fallback.
            half_w = max((marker_point_size / 72.0) * dx * 0.35, dx * 0.01)
            half_h = max((marker_point_size / 72.0) * dy * 0.35, dy * 0.01)
            _zoom_to_region(cx, cy, half_w, half_h)

        nid = node_ids[node_idx]
        if progressive:
            print(f"Zoomed to node {nid} (level {zoom_state['level']}). Press 'z' again to zoom further, 'r' to reset.")
        else:
            print(f"Zoomed to node {nid} area. Press 'z' to zoom further or 'r' to reset.")

    def _group_processes_by_stage(processes):
        """Group process labels by stage prefix like '[stage] name'.

        Returns an ordered dict-like list of (stage, set(processes)).
        """
        grouped = {}
        for p in sorted(set(processes)):
            if p.startswith('['):
                try:
                    end = p.index(']')
                    stage = p[1:end]
                    name = p[end+2:] if len(p) > end+2 else p[end+1:]
                except ValueError:
                    stage = 'Global'
                    name = p
            else:
                stage = 'Global'
                name = p

            grouped.setdefault(stage, set()).add(name)

        stage_order = {stage: index for index, stage in enumerate(stage_names)}
        ordered_stages = sorted(
            grouped,
            key=lambda stage: (
                stage != 'Global',
                stage_order.get(stage, len(stage_order)),
                stage.casefold(),
            ),
        )
        return {stage: grouped[stage] for stage in ordered_stages}

    def _format_grouped_processes(processes):
        if not processes:
            return '  None\n'
        grouped = _group_processes_by_stage(processes)
        lines = []
        for stage, procs in grouped.items():
            lines.append(f"  [{stage}]")
            for i, proc in enumerate(sorted(procs), 1):
                lines.append(f"    {i}. {proc}")
        return '\n'.join(lines) + '\n'

    # Process report viewer window (scroll + copy + export)
    report_state = {
        'fig': None,
        'text_ax': None,
        'status_ax': None,
        'title_ax': None,
        'body_artist': None,
        'status_artist': None,
        'title_artist': None,
        'copy_btn': None,
        'save_btn': None,
        'scroll_cid': None,
        'scroll_slider': None,
        'suppress_scroll_slider_callback': False,
        'lines': [],
        'start': 0,
        'page': 40,
        'full_text': '',
        'title': 'Process Report',
    }

    def _wrap_report_text(text, width=110):
        """Wrap report lines to keep them readable in the report panel."""
        wrapped = []
        for line in (text or '').splitlines():
            # Keep indentation for grouped process lines.
            indent_len = len(line) - len(line.lstrip(' '))
            indent = ' ' * indent_len
            content = line[indent_len:]

            if not content:
                wrapped.append('')
                continue

            chunks = textwrap.wrap(
                content,
                width=max(20, width - indent_len),
                break_long_words=False,
                break_on_hyphens=False,
                initial_indent=indent,
                subsequent_indent=indent + '  '
            )
            wrapped.extend(chunks if chunks else [indent])
        return wrapped

    def _copy_to_clipboard(text):
        # Try tkinter first (works for many backends/platforms).
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
            print('Report copied to clipboard.')
            return True
        except Exception:
            pass

        # Fallback to OS clipboard commands.
        try:
            if os.name == 'nt':
                subprocess.run(['clip'], input=text, text=True, check=True)
                print('Report copied to clipboard.')
                return True
            if sys.platform == 'darwin':
                subprocess.run(['pbcopy'], input=text, text=True, check=True)
                print('Report copied to clipboard.')
                return True
            subprocess.run(['xclip', '-selection', 'clipboard'], input=text, text=True, check=True)
            print('Report copied to clipboard.')
            return True
        except Exception as e:
            print(f'Clipboard copy not available: {e}')
            return False

    def _export_report_to_file(text):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'mdpa_report_{ts}.txt'
        primary_dir = mdpa_dir
        fallback_dir = os.getcwd()

        targets = [primary_dir]
        if os.path.normcase(fallback_dir) != os.path.normcase(primary_dir):
            targets.append(fallback_dir)

        last_error = None
        for target_dir in targets:
            try:
                os.makedirs(target_dir, exist_ok=True)
                out_path = os.path.join(target_dir, filename)
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                print(f'Report saved to {out_path}')
                return
            except Exception as e:
                last_error = e

        print(f'Failed to save report: {last_error}')

    def _render_report_body():
        rs = report_state
        if rs['fig'] is None or rs['body_artist'] is None:
            return
        total = len(rs['lines'])
        if total == 0:
            body = '(No content)'
            visible_start = 0
            visible_end = 0
        else:
            start = max(0, min(rs['start'], max(total - rs['page'], 0)))
            rs['start'] = start
            end = min(total, start + rs['page'])
            body = '\n'.join(rs['lines'][start:end])
            visible_start = start + 1
            visible_end = end
        rs['body_artist'].set_text(body)
        rs['title_artist'].set_text(rs['title'])

        if total == 0:
            status = 'No report content.'
        elif total <= rs['page']:
            status = f'All {total} lines shown.'
        else:
            status = (
                f'Lines {visible_start}-{visible_end} of {total}. '
                'Scroll with the mouse wheel or drag the scrollbar.'
            )
        rs['status_artist'].set_text(status)

        slider = rs.get('scroll_slider')
        if slider is not None:
            max_start = max(total - rs['page'], 0)
            slider_max = max(1, max_start)
            # Keep slider bounds aligned with current content length while
            # avoiding a zero-width range that Matplotlib warns about.
            slider.valmax = slider_max
            if hasattr(slider, 'ax'):
                slider.ax.set_ylim(0, slider_max)
            rs['start'] = max(0, min(rs['start'], max_start))
            rs['suppress_scroll_slider_callback'] = True
            slider.set_val(slider_max - rs['start'])
            rs['suppress_scroll_slider_callback'] = False

        rs['fig'].canvas.draw_idle()

    def _ensure_report_window():
        rs = report_state
        if rs['fig'] is not None and plt.fignum_exists(rs['fig'].number):
            return

        rfig = plt.figure(figsize=(9, 10))
        rfig.canvas.manager.set_window_title('MDPA Process Report')

        title_ax = rfig.add_axes([0.03, 0.93, 0.94, 0.05])
        title_ax.axis('off')
        title_artist = title_ax.text(0.0, 0.7, 'Process Report', fontsize=11, weight='bold', va='center')

        text_ax = rfig.add_axes([0.03, 0.10, 0.90, 0.82])
        text_ax.axis('off')
        body_artist = text_ax.text(0.0, 1.0, '', va='top', family='monospace', fontsize=8)

        status_ax = rfig.add_axes([0.03, 0.075, 0.90, 0.02])
        status_ax.axis('off')
        status_artist = status_ax.text(0.0, 0.5, '', va='center', fontsize=8, color='0.35')

        scroll_ax = rfig.add_axes([0.94, 0.10, 0.025, 0.82])
        scroll_ax.set_facecolor('#f2f2f2')
        scroll_slider = Slider(
            ax=scroll_ax,
            label='',
            valmin=0,
            valmax=1,
            valinit=0,
            valstep=1,
            orientation='vertical'
        )
        if hasattr(scroll_slider, 'valtext'):
            scroll_slider.valtext.set_visible(False)

        copy_ax = rfig.add_axes([0.03, 0.02, 0.12, 0.05])
        save_ax = rfig.add_axes([0.17, 0.02, 0.12, 0.05])
        copy_btn = Button(copy_ax, 'Copy')
        save_btn = Button(save_ax, 'Export')

        def _on_copy(_event):
            _copy_to_clipboard(report_state['full_text'])

        def _on_save(_event):
            _export_report_to_file(report_state['full_text'])

        def _on_scroll_slider(val):
            if report_state.get('suppress_scroll_slider_callback', False):
                return
            max_start = max(len(report_state['lines']) - report_state['page'], 0)
            slider_max = max(1, max_start)
            report_state['start'] = max(
                0, min(max_start, int(round(slider_max - val)))
            )
            _render_report_body()

        def _on_scroll(event):
            if event.inaxes != text_ax:
                return
            step = 4
            if event.button == 'up':
                report_state['start'] = max(0, report_state['start'] - step)
            elif event.button == 'down':
                report_state['start'] = min(max(len(report_state['lines']) - report_state['page'], 0), report_state['start'] + step)
            _render_report_body()

        copy_btn.on_clicked(_on_copy)
        save_btn.on_clicked(_on_save)
        scroll_slider.on_changed(_on_scroll_slider)
        scroll_cid = rfig.canvas.mpl_connect('scroll_event', _on_scroll)

        rs['fig'] = rfig
        rs['text_ax'] = text_ax
        rs['status_ax'] = status_ax
        rs['title_ax'] = title_ax
        rs['body_artist'] = body_artist
        rs['status_artist'] = status_artist
        rs['title_artist'] = title_artist
        # Keep widget references alive; otherwise callbacks can stop working.
        rs['copy_btn'] = copy_btn
        rs['save_btn'] = save_btn
        rs['scroll_slider'] = scroll_slider
        rs['scroll_cid'] = scroll_cid

    def _show_report(title, text):
        report_state['title'] = title
        report_state['full_text'] = text
        report_state['lines'] = _wrap_report_text(text, width=110)
        report_state['start'] = 0
        _ensure_report_window()
        _render_report_body()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='FigureCanvasAgg is non-interactive, and thus cannot be shown'
            )
            report_state['fig'].show()

    def _handle_node_click(event):
        # Ignore clicks outside canvas
        if event.inaxes is None and not (event.x is not None and event.y is not None):
            return
        
        # Click is on plot - find closest node
        if event.inaxes != ax:
            return
        
        click_pos = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(node_coords[:, :2] - click_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Only process if close enough (tolerance in data units)
        # Use a tolerance proportional to the data span
        tol = 0.01 * max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])
        if distances[closest_idx] > tol:
            return
        last_selected['idx'] = closest_idx
        if zoom_state['node_idx'] != closest_idx:
            zoom_state['node_idx'] = None
            zoom_state['half_width'] = None
            zoom_state['half_height'] = None
            zoom_state['level'] = 0
        
        nid = node_ids[closest_idx]
        coincident_ids = sorted(xy_to_node_ids.get(node_xy[nid], [nid]))
        containing_all = [name for name, nset in submodel_nodes.items() if nid in nset]
        # Report what is currently visible on plot first (same logic as rendering).
        containing = [name for name in containing_all if name not in hidden_submodels]
        if not containing:
            containing = containing_all
        msg = f"\nNode {nid} @ ({node_coords[closest_idx, 0]:.2f}, {node_coords[closest_idx, 1]:.2f})\n"
        msg += f"Coincident node ids (same XY): {', '.join(str(i) for i in coincident_ids)}\n"
        msg += f"SubModelParts: {', '.join(containing) or 'None'}\n"

        # If double-click: zoom to this node's small neighborhood
        if getattr(event, 'dblclick', False):
            _zoom_to_node_index(closest_idx)

        # For nodes that belong to multiple SubModelParts, print separate process
        # information for each SubModelPart instead of aggregating them together.
        if not containing:
            msg += "Processes:\n  None\n"
            print(msg)
            _show_report(f'Node {nid}', msg)
            return

        for idx, sub_name in enumerate(containing, start=1):
            procs = _get_processes_for_submodel(sub_name)

            ncount = len(submodel_nodes.get(sub_name, set()))
            msg += '\n' + ('-' * 40) + '\n'
            msg += f"SubModelPart {idx}: {sub_name}\n"
            msg += f"Total nodes: {ncount}\n"
            msg += f"Processes:\n"
            msg += _format_grouped_processes(procs)

        # Add a concise membership summary for all coincident nodes.
        if len(coincident_ids) > 1:
            msg += '\n' + ('=' * 40) + '\n'
            msg += 'Coincident node membership summary:\n'
            for other_nid in coincident_ids:
                other_subs = node_to_subs.get(other_nid, [])
                msg += f"- Node {other_nid}: {len(other_subs)} SubModelParts"
                if other_subs:
                    msg += f" -> {', '.join(other_subs)}"
                msg += '\n'

        print(msg)
        _show_report(f'Node {nid}', msg)

    # Left-mouse drag panning (touchpad press+move translates to left-button drag events).
    pan_state = {
        'active': False,
        'moved': False,
        'press_x': None,
        'press_y': None,
        'press_xlim': None,
        'press_ylim': None,
        'last_draw_time': 0.0,
    }

    def _toolbar_mode_active():
        """Return True when the matplotlib toolbar has an active interaction mode."""
        manager = getattr(fig.canvas, 'manager', None)
        toolbar = getattr(manager, 'toolbar', None)
        mode = getattr(toolbar, 'mode', '')
        return bool(mode)

    def _on_mouse_press(event):
        if event.button != 1:
            return
        if _toolbar_mode_active():
            return
        if event.inaxes != ax:
            return
        if event.x is None or event.y is None:
            return

        pan_state['active'] = True
        pan_state['moved'] = False
        pan_state['press_x'] = event.x
        pan_state['press_y'] = event.y
        pan_state['press_xlim'] = ax.get_xlim()
        pan_state['press_ylim'] = ax.get_ylim()
        pan_state['last_draw_time'] = 0.0

    def _on_mouse_move(event):
        if not pan_state['active']:
            return
        # Allow continuing drag even if cursor leaves the axes during fast motion.
        if event.x is None or event.y is None:
            return
        if _toolbar_mode_active():
            return

        press_xlim = pan_state['press_xlim']
        press_ylim = pan_state['press_ylim']
        if press_xlim is None or press_ylim is None:
            return

        # Compute pan from initial press anchor in pixel space for stable movement.
        bbox = ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return

        dx_px = event.x - pan_state['press_x']
        dy_px = event.y - pan_state['press_y']

        dx_data = dx_px * (press_xlim[1] - press_xlim[0]) / bbox.width
        dy_data = dy_px * (press_ylim[1] - press_ylim[0]) / bbox.height

        ax.set_xlim(press_xlim[0] - dx_data, press_xlim[1] - dx_data)
        ax.set_ylim(press_ylim[0] - dy_data, press_ylim[1] - dy_data)

        # Throttle redraw a bit to keep drag responsive for large models.
        now = perf_counter()
        if (now - pan_state['last_draw_time']) >= (1.0 / 120.0):
            fig.canvas.draw_idle()
            pan_state['last_draw_time'] = now

        move_px = abs(dx_px) + abs(dy_px)
        if move_px >= 2:
            pan_state['moved'] = True

    def _on_mouse_release(event):
        if event.button != 1:
            return

        was_active = pan_state['active']
        was_drag = pan_state['moved']
        pan_state['active'] = False
        pan_state['moved'] = False
        pan_state['press_x'] = None
        pan_state['press_y'] = None
        pan_state['press_xlim'] = None
        pan_state['press_ylim'] = None
        pan_state['last_draw_time'] = 0.0

        if was_active and was_drag:
            fig.canvas.draw_idle()

        # Treat release as a click only when no drag happened.
        if was_active and not was_drag:
            _handle_node_click(event)

    fig.canvas.mpl_connect('button_press_event', _on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', _on_mouse_move)
    fig.canvas.mpl_connect('button_release_event', _on_mouse_release)
    
    # Add keyboard event handler for number keys to show processes
    key_map = {
        '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
        '6': 5, '7': 6, '8': 7, '9': 8, '0': 9,
    }

    def on_key(event):
        """Handle keyboard input to show processes for SubModelParts."""
        if event.key is None:
            return
        
        if event.key in key_map:
            idx = key_map[event.key]
            if idx < len(subnames):
                subpart_name = subnames[idx]
                procs = _get_processes_for_submodel(subpart_name)
                ncount = len(submodel_nodes.get(subpart_name, set()))
                msg = f"\n{'='*60}\n"
                msg += f"SubModelPart #{idx+1}: {subpart_name}\n"
                msg += f"{'='*60}\n"
                msg += f"Total nodes: {ncount}\n"
                msg += f"Processes:\n"
                msg += _format_grouped_processes(procs)
                print(msg)
                _show_report(f'SubModelPart #{idx+1}: {subpart_name}', msg)
            else:
                print(f"SubModelPart #{event.key} not found (only {len(subnames)} available)")

        if event.key == 'z':
            if last_selected['idx'] is None:
                print("No node selected. Click a node first, then press 'z' to zoom.")
            else:
                _zoom_to_node_index(last_selected['idx'], progressive=True)

        # Reset view
        if event.key == 'r':
            ax.set_xlim(orig_xlim)
            ax.set_ylim(orig_ylim)
            zoom_state['node_idx'] = None
            zoom_state['half_width'] = None
            zoom_state['half_height'] = None
            zoom_state['level'] = 0
            fig.canvas.draw_idle()
            print("View reset to full extent.")
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Do not call tight_layout() here: it can resize axes and overlap with
    # manually positioned control panels.
    plt.show()

