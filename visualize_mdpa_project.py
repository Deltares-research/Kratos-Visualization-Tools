#!/usr/bin/env python3
"""
====================================================================

⚠️  AI-ASSISTED PROTOTYPE — PROVENANCE & VALIDATION NOTICE

====================================================================

Author: Gennady Markelov

This contribution (scripts and documentation) was generated with AI assistance and reviewed on selected internal test cases. It is provided as a prototype and should not be considered production-ready.
Please perform independent review, validation, and testing before use.
 
====================================================================

Lightweight MDPA visualizer using matplotlib (no heavy VTK dependencies).

Features:
- Parse Kratos `.mdpa` file for Nodes and SubModelPart definitions
- Visualize nodes colored by SubModelPart membership in 2D scatter plot
- Click on nodes to see SubModelPart info and associated processes
- Much lighter than pyvista version (matplotlib only)
- Separated Faces leaves nodes fixed when visible interface directions conflict;
  hiding an interface recalculates display-only offsets without changing the MDPA.

Usage:
    python visualize_mdpa_project.py path/to/model.mdpa [--project params.json]
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

from kratos_visualization_tools._compat import install_facade as _install_facade
from kratos_visualization_tools import mdpa as _mdpa
from kratos_visualization_tools import project as _project
from kratos_visualization_tools import visualization as _visualization
from kratos_visualization_tools import cli as _cli

_install_facade(
    _sys.modules[__name__],
    {
        "argparse": _cli,
        "colorsys": _visualization,
        "json": _project,
        "os": _project,
        "re": _project,
        "subprocess": _visualization,
        "sys": _visualization,
        "textwrap": _visualization,
        "warnings": _visualization,
        "dataclass": _mdpa,
        "Enum": _mdpa,
        "perf_counter": _visualization,
        "datetime": _visualization,
        "defaultdict": _mdpa,
        "product": _mdpa,
        "plt": _visualization,
        "Path": _visualization,
        "Button": _visualization,
        "CheckButtons": _visualization,
        "Slider": _visualization,
        "TextBox": _visualization,
        "np": _mdpa,
        "GeometryType": _mdpa,
        "ENTITY_2D3N_GEOMETRIES": _mdpa,
        "_normalize_submodelpart_name": _mdpa,
        "_stage_from_process_label": _mdpa,
        "_resolve_constraint_relations": _mdpa,
        "_connection_paths": _mdpa,
        "_normal_segments": _mdpa,
        "_separation_paths": _mdpa,
        "ParsedMdpa": _mdpa,
        "_normal_vector": _mdpa,
        "parse_mdpa_data": _mdpa,
        "_build_face_separation": _mdpa,
        "_separated_coordinates": _mdpa,
        "parse_mdpa": _mdpa,
        "_resolve_stage_activity": _project,
        "parse_project_parameters": _project,
        "find_mdpa_from_project": _project,
        "visualize_mdpa": _visualization,
        "main": _cli,
    },
    (_mdpa, _project, _visualization, _cli),
)

if __name__ == "__main__":
    main()
