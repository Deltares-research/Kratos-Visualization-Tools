"""
Visualize GiD .post.res results on GiD .post.msh meshes, with profile focus.

Features:
- Parse GiD mesh (.post.msh): nodes, triangle/quad elements, line meshes
- Parse GiD results (.post.res): Scalar/Vector OnNodes blocks
- List available result names/steps
- Plot domain map + profile for a selected result

Examples:
  python Eemdijk/tools/visualize_gid_results.py       Eemdijk/20_Bovenbelasting       --list

  python Eemdijk/tools/visualize_gid_results.py       Eemdijk/20_Bovenbelasting       --result WATER_PRESSURE --step latest

  python Eemdijk/tools/visualize_gid_results.py       Eemdijk/20_Bovenbelasting       --result REACTION --component magnitude --step latest
"""

import sys as _sys
from .._compat import install_facade as _install_facade
from . import models as _models
from . import calculations as _calculations
from . import parsing as _parsing
from . import results as _results
from . import exports as _exports
from . import drawing as _drawing
from . import plotting as _plotting
from . import viewer as _viewer
from . import comparison as _comparison
from . import config as _config
from . import cli as _cli

_install_facade(
    _sys.modules[__name__],
    {
        "argparse": _cli,
        "csv": _exports,
        "json": _exports,
        "math": _calculations,
        "re": _models,
        "textwrap": _comparison,
        "defaultdict": _calculations,
        "dataclass": _models,
        "Path": _models,
        "Dict": _models,
        "Iterable": _models,
        "List": _models,
        "Optional": _models,
        "Sequence": _models,
        "Set": _models,
        "Tuple": _models,
        "plt": _plotting,
        "np": _calculations,
        "mtri": _calculations,
        "animation": _viewer,
        "Normalize": _comparison,
        "TextBox": _viewer,
        "RESULT_HEADER_RE": _models,
        "MESH_HEADER_RE": _models,
        "RESULT_LOCATION_ON_NODES": _models,
        "RESULT_LOCATION_ON_GAUSS_POINTS": _models,
        "VECTOR_COMPONENTS": _models,
        "TENSOR_COMPONENTS": _models,
        "MeshBlock": _models,
        "ResultHeader": _models,
        "ActiveSubmodelPart": _models,
        "SolutionData": _models,
        "ViewState": _models,
        "cycle_sequence": _calculations,
        "validate_component": _calculations,
        "build_output_path": _exports,
        "build_result_export_path": _exports,
        "compare_result_names": _results,
        "calculate_shared_scale": _calculations,
        "interpolate_scalar_field": _calculations,
        "calculate_symmetric_scale": _calculations,
        "draw_scalar_isolines": _drawing,
        "draw_triangulation_boundary": _drawing,
        "parse_post_msh": _parsing,
        "resolve_active_nodes": _parsing,
        "scan_result_headers": _parsing,
        "scan_gauss_point_counts": _parsing,
        "load_values": _parsing,
        "project_element_to_nodes": _calculations,
        "find_best_location": _results,
        "pick_step": _results,
        "vector_component": _calculations,
        "tensor_component": _calculations,
        "field_scalar_value": _calculations,
        "reduce_result_value": _calculations,
        "build_scalar_field_from_values": _calculations,
        "finite_difference_gradient": _calculations,
        "compute_nodal_gradient": _calculations,
        "detect_profile_nodes": _parsing,
        "parse_submodelpart_entities": _parsing,
        "resolve_default_mdpa_path": _results,
        "resolve_mesh_result_paths": _results,
        "discover_solution_paths": _results,
        "load_solution_data": _results,
        "load_result_values": _results,
        "load_solution_field": _results,
        "build_node_arrays": _calculations,
        "sample_vertical_profile_from_triangulation": _calculations,
        "sample_profile_path_from_triangulation": _calculations,
        "triangulation_from_connectivity": _calculations,
        "InteractiveViewer": _viewer,
        "ComparisonViewer": _comparison,
        "render_plot_static": _plotting,
        "render_plot_interactive": _plotting,
        "render_comparison": _plotting,
        "print_result_summary": _results,
        "print_comparison_summary": _results,
        "_parse_line_spec": _exports,
        "_sample_line_values_idw": _exports,
        "export_line_csv_from_json": _exports,
        "parse_profile_path": _config,
        "load_visualization_settings": _config,
        "apply_visualization_settings": _config,
        "main": _cli,
    },
    (_models, _calculations, _parsing, _results, _exports, _drawing, _plotting, _viewer, _comparison, _config, _cli,),
)
