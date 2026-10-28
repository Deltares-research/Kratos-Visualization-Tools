"""Project-parameter and orchestration JSON parsing."""
import json
import os
import re
from collections import defaultdict

from .mdpa import _normalize_submodelpart_name, _stage_from_process_label


def parse_project_parameters(paths, base_dir=None):
    """Extract processes/operations from ProjectParameters or orchestrator JSON files."""
    mapping = defaultdict(list)
    stage_names = []
    seen_stages = set()
    orchestrator_stage_names = []
    standalone_stage_names = []
    stage_referenced_submodels = defaultdict(set)

    def _register_stage(stage_name, source=None):
        if isinstance(stage_name, str) and stage_name and stage_name not in seen_stages:
            seen_stages.add(stage_name)
            stage_names.append(stage_name)
        if source == 'orchestrator' and stage_name not in orchestrator_stage_names:
            orchestrator_stage_names.append(stage_name)
        elif source == 'standalone' and stage_name not in standalone_stage_names:
            standalone_stage_names.append(stage_name)

    def _standalone_stage_sort_key(stage_name):
        match = re.search(r'(\d+)$', stage_name)
        if match:
            return (0, int(match.group(1)), stage_name.casefold())
        return (1, 0, stage_name.casefold())

    def resolve_project_path(path):
        """Resolve project JSON path.

        Relative paths are interpreted against the mdpa directory first,
        then against the current working directory as a fallback.
        """
        # Be tolerant to accidental quotes in CLI args, e.g. -p file.json'
        raw = (path or '').strip()
        expanded = os.path.expanduser(raw.strip("\"'"))

        if os.path.isabs(expanded):
            return expanded if os.path.isfile(expanded) else None

        if base_dir:
            candidate = os.path.normpath(os.path.join(base_dir, expanded))
            if os.path.isfile(candidate):
                return candidate

        if os.path.isfile(expanded):
            return expanded

        return None

    def standalone_stage_name(path, project_data):
        """Return a stage name for standalone ProjectParameters_stage files."""
        if not isinstance(project_data, dict) or 'stages' in project_data:
            return None
        stem = os.path.splitext(os.path.basename(path))[0]
        match = re.match(r'^ProjectParameters_?(.+)$', stem, re.IGNORECASE)
        return match.group(1) if match else None

    def extract_submodelpart_names(obj, collection=None):
        """Recursively extract SubModelPart names from various fields."""
        if collection is None:
            collection = []
        
        if isinstance(obj, dict):
            # Check for model_part_name (single)
            if 'model_part_name' in obj:
                mp = obj['model_part_name']
                if isinstance(mp, str):
                    collection.append(mp)
            
            # Check for model_part_name_list (array)
            if 'model_part_name_list' in obj:
                mp_list = obj['model_part_name_list']
                if isinstance(mp_list, list):
                    collection.extend(mp_list)
            
            # Recurse into all values
            for v in obj.values():
                extract_submodelpart_names(v, collection)
        elif isinstance(obj, list):
            for item in obj:
                extract_submodelpart_names(item, collection)
        
        return collection

    def extract_processes(obj, collection=None, stage_name=None, stage_source=None):
        """Extract processes/operations with their model parts."""
        if collection is None:
            collection = []

        def build_stage_process_label(process_label, stage):
            if stage:
                return f"[{stage}] {process_label}"
            return process_label

        def build_process_label(process_data, fallback_name):
            params = process_data.get('Parameters', {}) if isinstance(process_data, dict) else {}
            process_name = (
                process_data.get('process_name')
                or process_data.get('python_module')
                or process_data.get('name')
                or fallback_name
            )
            variable_name = params.get('variable_name')
            if variable_name:
                return f"{process_name} ({variable_name})"
            variable_names = params.get('variable_names')
            if isinstance(variable_names, list):
                filtered = [v for v in variable_names if isinstance(v, str)]
                if filtered:
                    return f"{process_name} ({', '.join(filtered)})"
            return process_name

        def extract_process_model_parts(params):
            model_parts = []

            mp = params.get('model_part_name') or params.get('model_part')
            if isinstance(mp, str):
                model_parts.append(mp)

            # AssignMasterSlaveConstraintsToNeighboursUtility specific fields.
            master_mp = params.get('master_model_part_name')
            if isinstance(master_mp, str):
                model_parts.append(master_mp)

            slave_mp = params.get('slave_model_part_name')
            if isinstance(slave_mp, str):
                model_parts.append(slave_mp)

            mp_list = params.get('model_part_name_list')
            if isinstance(mp_list, list):
                model_parts.extend([name for name in mp_list if isinstance(name, str)])

            return model_parts

        def process_stage_data(stage_key, stage_data):
            _register_stage(stage_key, source=stage_source or 'orchestrator')
            if not isinstance(stage_data, dict):
                return

            # Look in stage_preprocess operations/modelers.
            if 'stage_preprocess' in stage_data:
                preproc = stage_data['stage_preprocess']
                if isinstance(preproc, dict):
                    # Extract from operations
                    if 'operations' in preproc:
                        ops = preproc['operations']
                        if isinstance(ops, list):
                            for op in ops:
                                if isinstance(op, dict):
                                    op_name = op.get('name', 'Operation')
                                    op_name = build_stage_process_label(op_name, stage_key)
                                    mp_list = extract_submodelpart_names(op)
                                    for mp in mp_list:
                                        collection.append((mp, op_name))

                    # Extract from modelers
                    if 'modelers' in preproc:
                        mods = preproc['modelers']
                        if isinstance(mods, list):
                            for mod in mods:
                                if isinstance(mod, dict):
                                    mod_name = mod.get('name', 'Modeler')
                                    mod_name = build_stage_process_label(mod_name, stage_key)
                                    mp_list = extract_submodelpart_names(mod)
                                    for mp in mp_list:
                                        collection.append((mp, mod_name))

            extract_processes(
                stage_data,
                collection,
                stage_name=stage_key,
                stage_source=stage_source,
            )

            # Also collect solver settings usage from either supported layout.
            solver_settings = stage_data.get('solver_settings') if isinstance(stage_data, dict) else None
            if not isinstance(solver_settings, dict):
                stage_settings = stage_data.get('stage_settings', {}) if isinstance(stage_data, dict) else {}
                solver_settings = stage_settings.get('solver_settings', {}) if isinstance(stage_settings, dict) else {}
            if isinstance(solver_settings, dict):
                prob_list = solver_settings.get('problem_domain_sub_model_part_list')
                body_list = solver_settings.get('body_domain_sub_model_part_list')
                process_list = solver_settings.get('processes_sub_model_part_list')
                prob_set = set(p for p in prob_list if isinstance(p, str)) if isinstance(prob_list, list) else set()
                body_set = set(p for p in body_list if isinstance(p, str)) if isinstance(body_list, list) else set()
                process_set = set(p for p in process_list if isinstance(p, str)) if isinstance(process_list, list) else set()

                if prob_set or body_set:
                    # If both lists are identical, only report as problem_domain to avoid duplication
                    if prob_set and prob_set == body_set:
                        lbl = build_stage_process_label('Used as problem_domain', stage_key)
                        for mp in sorted(prob_set):
                            collection.append((mp, lbl))
                    else:
                        # Report problem_domain entries
                        if prob_set:
                            lblp = build_stage_process_label('Used as problem_domain', stage_key)
                            for mp in sorted(prob_set):
                                collection.append((mp, lblp))
                        # Report body_domain entries that are not already reported
                        extra_body = body_set - prob_set
                        if extra_body:
                            lblb = build_stage_process_label('Used as body_domain', stage_key)
                            for mp in sorted(extra_body):
                                collection.append((mp, lblb))
                if process_set:
                    lbl_processes = build_stage_process_label('Used by processes', stage_key)
                    for mp in sorted(process_set):
                        collection.append((mp, lbl_processes))
        
        if isinstance(obj, dict):
            # Old format: processes dict with name as key
            if 'processes' in obj and isinstance(obj['processes'], dict):
                for proc_name, proc_list in obj['processes'].items():
                    if isinstance(proc_list, list):
                        for p in proc_list:
                            if isinstance(p, dict):
                                params = p.get('Parameters', {})
                                process_label = build_process_label(p, proc_name)
                                process_label = build_stage_process_label(process_label, stage_name)
                                for mp in extract_process_model_parts(params):
                                    collection.append((mp, process_label))
            
            # Processes as list
            elif 'processes' in obj and isinstance(obj['processes'], list):
                for p in obj['processes']:
                    if isinstance(p, dict):
                        params = p.get('Parameters', {})
                        process_label = build_process_label(p, 'Process')
                        process_label = build_stage_process_label(process_label, stage_name)
                        for mp in extract_process_model_parts(params):
                            collection.append((mp, process_label))
            
            # Orchestrator/stages format: look for operations, modelers, etc
            if 'stages' in obj:
                stages = obj['stages']
                if isinstance(stages, dict):
                    for stage_key, stage_data in stages.items():
                        process_stage_data(stage_key, stage_data)
                elif isinstance(stages, list):
                    for idx, stage_data in enumerate(stages, start=1):
                        if isinstance(stage_data, dict):
                            stage_key = (
                                stage_data.get('name')
                                or stage_data.get('stage_name')
                                or stage_data.get('id')
                                or f"stage_{idx}"
                            )
                        else:
                            stage_key = f"stage_{idx}"
                        process_stage_data(stage_key, stage_data)
            
            # Recurse
            for k, v in obj.items():
                if k == 'stages':
                    continue
                extract_processes(v, collection, stage_name=stage_name)
        elif isinstance(obj, list):
            for item in obj:
                extract_processes(item, collection, stage_name=stage_name)
        
        return collection

    for p in paths:
        resolved_path = resolve_project_path(p)
        if resolved_path is None:
            print(f"Warning: Could not find project file '{p}'")
            continue

        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                j = json.load(f)
            
            # Standalone ProjectParameters_stage*.json files encode the stage
            # in their filename rather than in a top-level stages object.
            filename_stage = standalone_stage_name(resolved_path, j)
            if filename_stage:
                _register_stage(filename_stage, source='standalone')
            processes_list = extract_processes(j, stage_name=filename_stage)
            if filename_stage and isinstance(j.get('solver_settings'), dict):
                standalone_solver = {
                    'stages': {
                        filename_stage: {
                            'solver_settings': j['solver_settings']
                        }
                    }
                }
                processes_list.extend(
                    extract_processes(standalone_solver, stage_source='standalone')
                )
            
            # Map to SubModelParts using leaf name if parent prefixes are present.
            for mp, process_name in processes_list:
                sub_mp = _normalize_submodelpart_name(mp)
                mapping[sub_mp].append(process_name)

                # Track all SubModelParts referenced by each stage, including
                # processes, operations/modelers, and stage domain lists.
                stage = _stage_from_process_label(process_name)
                if stage != 'Global':
                    stage_referenced_submodels[stage].add(sub_mp)
        except Exception as e:
            print(f"Warning: Could not parse project file '{resolved_path}': {e}")

    # Fallback: if stages were only discovered through labels, expose them in selector.
    for stage in stage_referenced_submodels.keys():
        _register_stage(stage)

    # An orchestrator's stages block defines the authoritative execution order.
    # Standalone ProjectParameters_stage files use their numeric suffix instead.
    ordered_stages = list(orchestrator_stage_names)
    ordered_stages.extend(
        sorted(
            (stage for stage in standalone_stage_names if stage not in ordered_stages),
            key=_standalone_stage_sort_key,
        )
    )
    ordered_stages.extend(
        stage for stage in stage_names if stage not in ordered_stages
    )

    return mapping, ordered_stages, stage_referenced_submodels


def find_mdpa_from_project(project_path):
    """Return the MDPA path referenced by an ImportMDPAModeler."""
    with open(project_path, 'r', encoding='utf-8') as f:
        project_data = json.load(f)

    def find_input_filename(obj):
        if isinstance(obj, dict):
            if (
                obj.get('name') == 'Modelers.KratosMultiphysics.ImportMDPAModeler'
                and isinstance(obj.get('Parameters'), dict)
                and isinstance(obj['Parameters'].get('input_filename'), str)
            ):
                return obj['Parameters']['input_filename']
            for value in obj.values():
                filename = find_input_filename(value)
                if filename:
                    return filename
        elif isinstance(obj, list):
            for value in obj:
                filename = find_input_filename(value)
                if filename:
                    return filename
        return None

    input_filename = find_input_filename(project_data)
    if not input_filename:
        raise ValueError(
            "Could not find ImportMDPAModeler Parameters.input_filename"
        )

    expanded = os.path.expanduser(input_filename.strip("\"'"))
    if not os.path.splitext(expanded)[1]:
        expanded += '.mdpa'
    if os.path.isabs(expanded):
        return expanded

    project_dir = os.path.dirname(os.path.abspath(project_path))
    project_relative = os.path.normpath(os.path.join(project_dir, expanded))
    if os.path.isfile(project_relative):
        return project_relative

    return os.path.normpath(expanded)

