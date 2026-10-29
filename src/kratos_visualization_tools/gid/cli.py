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

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from .calculations import project_element_to_nodes
from .comparison import ComparisonViewer
from .config import apply_visualization_settings, load_visualization_settings, parse_profile_path
from .exports import export_line_csv_from_json
from .models import ActiveSubmodelPart
from .parsing import detect_profile_nodes, load_values, parse_post_msh, parse_submodelpart_entities, scan_result_headers
from .plotting import render_comparison, render_plot_interactive, render_plot_static
from .results import compare_result_names, discover_solution_paths, load_solution_data, pick_step, print_comparison_summary, print_result_summary, resolve_default_mdpa_path, resolve_mesh_result_paths


def main():
    parser = argparse.ArgumentParser(description="Visualize GiD .post.res fields on .post.msh mesh")
    parser.add_argument(
        "name",
        nargs="?",
        help=(
            "Base path/name or explicit mesh/result file path. "
            "Accepted pairs: <name>.post.msh + <name>.post.res or <name>.msh + <name>.res"
        ),
    )
    parser.add_argument("--compare", default="", help="Second GiD solution to compare with the positional solution")
    parser.add_argument("--folder", default="", help="Folder containing multiple GiD mesh/result solution pairs")
    parser.add_argument("--settings", default="", help="Load a saved visualization settings JSON file")
    parser.add_argument("--extract-json", default="", help="JSON input for line extraction to CSV")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path for --extract-json mode")
    parser.add_argument(
        "--plot-config",
        default="",
        help="plot_saved_results JSON config to plot after --extract-json has written its CSV",
    )
    parser.add_argument(
        "--plot-show",
        action="store_true",
        help="With --plot-config, show the plots in the viewer window instead of saving them",
    )
    parser.add_argument("--list", action="store_true", help="List available result blocks and exit")
    parser.add_argument("--result", default="WATER_PRESSURE", help="Result name, e.g. WATER_PRESSURE, REACTION, AXIAL_FORCE")
    parser.add_argument("--step", default="latest", help="Time step number or 'latest'")
    parser.add_argument("--location", default=None, help="Result location: OnNodes or OnGaussPoints (auto-detect if omitted)")
    parser.add_argument(
        "--component",
        default="magnitude",
        choices=["x", "y", "z", "magnitude"],
        help="Vector component or magnitude (ignored for scalar)",
    )
    parser.add_argument(
        "--tensor-component",
        default="trace",
        choices=["xx", "xy", "yx", "yy", "trace", "magnitude"],
        help="Matrix/tensor reduction used to plot matrix results",
    )
    parser.add_argument("--profile-x", type=float, default=18.0, help="Fallback profile x coordinate")
    parser.add_argument(
        "--profile-path",
        default=None,
        help="Profile segment/polyline as x,y;x,y;... (at least two points)",
    )
    parser.add_argument("--x-tol", type=float, default=0.35, help="Tolerance for vertical-line/profile detection")
    parser.add_argument("--active", default="", help="SubModelPart name to visualize exclusively")
    parser.add_argument("--mdpa", default="", help="Optional .mdpa path for --active filtering")
    parser.add_argument("--save", default="", help="Optional output image path; if omitted, use interactive mode")
    parser.add_argument("--interactive", action="store_true", help="Force interactive mode (default if --save omitted)")
    parser.add_argument("--gui", action="store_true", help="Use the optional PySide6 single-solution window (install .[gui])")
    parser.add_argument("--isolines", action="store_true", help="Show solid isolines over filled single-solution fields")
    parser.add_argument("--isoline-labels", action="store_true", help="Label isolines compactly with their values")
    parser.add_argument("--contour-levels", type=int, default=12, help="Number of isoline levels (minimum 2)")
    parser.add_argument("--profile", action="store_true", help="Show the vertical profile panel")
    parser.add_argument("--no-domain-edge", action="store_true", help="Hide grey solution-domain boundaries in comparison mode")
    parser.add_argument(
        "--colormap",
        choices=ComparisonViewer.colormaps,
        default=None,
        help="Color map for comparison filled fields and difference mode",
    )
    parser.add_argument(
        "--comparison-style",
        choices=ComparisonViewer.styles,
        default="background+isolines",
        help="Comparison overlay style: filled background and isolines, or the reverse",
    )
    parser.add_argument("--vmin", type=float, default=None, help="Shared comparison color scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Shared comparison color scale maximum")
    parser.add_argument(
        "--inspection-method",
        choices=["nearest", "idw"],
        default="nearest",
        help="Coordinate inspection method in comparison mode: nearest node or inverse-distance weighting",
    )

    args = parser.parse_args()

    def validate_gui():
        if args.gui:
            for incompatible in ("compare", "list", "extract_json"):
                if getattr(args, incompatible):
                    parser.error(f"--gui cannot be used with --{incompatible.replace('_', '-')}")
            if args.save and not args.interactive:
                parser.error("--gui cannot be used with static --save; add --interactive")
            if args.vmin is not None or args.vmax is not None:
                parser.error("--gui manual color ranges are not yet supported; omit --vmin/--vmax")

    validate_gui()
    if (args.plot_config or args.plot_show) and not args.extract_json:
        parser.error("--plot-config and --plot-show require --extract-json")
    if args.plot_show and not args.plot_config:
        parser.error("--plot-show requires --plot-config")

    args.initial_show_gradient = False
    args.initial_show_mesh = False
    args.initial_colormap = None
    args.initial_show_deformation = False
    args.initial_deformation_scale = 1.0
    solution_paths = None
    if args.settings:
        settings_path = Path(args.settings).expanduser().resolve()
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        settings = load_visualization_settings(settings_path)
        apply_visualization_settings(args, settings)
        if settings["mode"] == "comparison" and not args.compare:
            solutions = settings.get("solutions")
            if not isinstance(solutions, list) or len(solutions) != 2:
                raise ValueError("Comparison settings must contain two solutions.")
            first_solution, second_solution = solutions
            if not isinstance(first_solution, dict) or not isinstance(second_solution, dict):
                raise ValueError("Comparison settings solutions must be objects.")
            args.name = first_solution.get("mesh")
            args.compare = second_solution.get("mesh", "")
        elif settings["mode"] == "single" and args.compare:
            raise ValueError("Single-solution settings cannot be used with --compare.")

    validate_gui()
    if args.extract_json:
        json_path = Path(args.extract_json).expanduser().resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"Extraction JSON not found: {json_path}")
        csv_path = Path(args.csv_out).expanduser().resolve() if args.csv_out else None
        out_csv = export_line_csv_from_json(json_path, csv_out=csv_path)
        print(f"Saved CSV: {out_csv}")
        if args.plot_config:
            plot_config_path = Path(args.plot_config).expanduser().resolve()
            if not plot_config_path.exists():
                raise FileNotFoundError(f"Plot config not found: {plot_config_path}")
            # Imported here because plot_saved_results switches matplotlib to the Agg backend.
            from .. import saved_results as plot_saved_results

            plot_argv = [str(plot_config_path)] + (["--show"] if args.plot_show else [])
            if plot_saved_results.main(plot_argv) != 0:
                raise RuntimeError(f"Plotting with {plot_config_path} failed")
        return

    if args.folder:
        if args.compare:
            parser.error("--folder cannot be combined with --compare")
        folder_path = Path(args.folder).expanduser().resolve()
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Solution folder not found: {folder_path}")
        solution_paths = discover_solution_paths(folder_path)
        if not solution_paths:
            raise RuntimeError(f"No mesh/result solution pairs found in {folder_path}")
        args.name = str(solution_paths[0][0])

    if not args.name:
        parser.error("name is required unless --extract-json is used or comparison settings provide solution paths")

    profile_path = parse_profile_path(args.profile_path)

    if args.compare:
        first = load_solution_data(args.name, "solution 1", base_dir=Path.cwd())
        second = load_solution_data(args.compare, "solution 2", base_dir=Path.cwd())
        print_comparison_summary(first, second)
        common_names, _, _ = compare_result_names(first.headers, second.headers)
        if not common_names:
            raise RuntimeError("The two solutions have no result variables in common.")
        if args.list:
            return

        if args.result not in common_names:
            print(f"Requested result '{args.result}' is not common; using '{common_names[0]}'.")
            args.result = common_names[0]

        use_interactive = (not args.save) or args.interactive
        print("Launching GiD solution comparison...")
        print("Comparison controls: [n/p]=result, [o]=overlay style, [a]=automatic scale, [/]=scale, [w]=inspection method, [q]=quit")
        render_comparison(
            first=first,
            second=second,
            initial_result=args.result,
            initial_step=args.step,
            initial_location=args.location,
            component=args.component,
            tensor_component_mode=args.tensor_component,
            profile_x=args.profile_x,
            style=args.comparison_style,
            vmin=args.vmin,
            vmax=args.vmax,
            out_path=None if use_interactive else Path(args.save),
            inspection_method=args.inspection_method,
            show_profile=args.profile,
            show_domain_edge=not args.no_domain_edge,
            colormap=args.colormap,
            contour_levels=args.contour_levels,
            show_isoline_labels=args.isoline_labels,
            profile_path=profile_path,
        )
        return

    if solution_paths is None:
        msh_path, res_path = resolve_mesh_result_paths(args.name, base_dir=Path.cwd())
    else:
        if args.save and not args.interactive:
            parser.error("Folder mode requires interactive mode so solutions can be switched")
        msh_path, res_path = solution_paths[0]

    if not msh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {msh_path}")
    if not res_path.exists():
        raise FileNotFoundError(f"Result file not found: {res_path}")

    headers = scan_result_headers(res_path)
    if not headers:
        raise RuntimeError(f"No parsable Result blocks found in {res_path}")

    if args.list:
        print_result_summary(headers)
        return

    all_nodes, triangles, surface_elements, line_meshes, _ = parse_post_msh(msh_path)

    profile_nodes = detect_profile_nodes(
        all_nodes=all_nodes,
        line_meshes=line_meshes,
        profile_x=args.profile_x,
        x_tol=args.x_tol,
    )

    active_filter: Optional[ActiveSubmodelPart] = None
    if args.active:
        if args.mdpa:
            mdpa_path = Path(args.mdpa)
        else:
            auto_mdpa = resolve_default_mdpa_path(args.name)
            if auto_mdpa is None:
                raise FileNotFoundError(
                    "Could not auto-locate .mdpa for --active filter. Provide --mdpa explicitly."
                )
            mdpa_path = auto_mdpa

        if not mdpa_path.exists():
            raise FileNotFoundError(f"MDPA file not found: {mdpa_path}")

        active_filter = parse_submodelpart_entities(mdpa_path, args.active)
        if not active_filter.node_ids and not active_filter.entity_ids:
            print(
                f"Warning: submodelpart '{args.active}' not found or empty in {mdpa_path}. "
                "No active filtering will be applied."
            )
            active_filter = None
        else:
            print(
                f"Active filter: submodelpart='{active_filter.name}', "
                f"nodes={len(active_filter.node_ids)}, entities={len(active_filter.entity_ids)}"
            )

    if (
        active_filter is not None
        and args.location is None
        and active_filter.entity_ids
        and any(h.name == args.result and h.location == "OnGaussPoints" for h in headers)
    ):
        args.location = "OnGaussPoints"
        print("Auto-selected location=OnGaussPoints for active condition/element-based submodelpart filter.")

    if not profile_nodes:
        print(
            "Warning: profile nodes could not be auto-detected. "
            "Try --profile-x and --x-tol tuned for your model."
        )

    # Use interactive mode unless --save is specified
    use_interactive = (not args.save) or args.interactive

    if use_interactive:
        renderer = render_plot_interactive
        if args.gui:
            try:
                from .qt_viewer import render_plot_gui
            except ModuleNotFoundError as exc:
                if exc.name != "PySide6" and not (exc.name or "").startswith("PySide6."):
                    raise
                parser.error("The Qt viewer requires PySide6. Install it with: pip install '.[gui]'")
            renderer = render_plot_gui
        print("Launching interactive viewer...")
        if not args.gui:
            print("Keyboard shortcuts: [n/p]=next/prev result, [j/k]=prev/next step, [x/y/z/g]=component, [t]=tensor reduction, [l]=isolines on/off, [m]=mesh on/off, [+/-/r]=zoom, right-drag or Shift+left-drag=pan, [i]=typed inspect, mouse-left=inspect, [?]=list, [q]=quit")
        renderer(
            headers=headers,
            res_path=res_path,
            all_nodes=all_nodes,
            triangles=triangles,
            surface_elements=surface_elements,
            line_meshes=line_meshes,
            profile_nodes=profile_nodes,
            initial_result=args.result,
            initial_step=args.step,
            initial_location=args.location,
            profile_x=args.profile_x,
            x_tol=args.x_tol,
            initial_tensor_component=args.tensor_component,
            active_filter=active_filter,
            show_isolines=args.isolines,
            show_profile=args.profile,
            contour_levels=args.contour_levels,
            show_isoline_labels=args.isoline_labels,
            initial_component=args.component,
            initial_show_gradient=args.initial_show_gradient,
            initial_show_mesh=args.initial_show_mesh,
            initial_colormap=(args.colormap or args.initial_colormap) if args.gui else args.initial_colormap,
            profile_path=profile_path,
            solution_paths=solution_paths,
            initial_show_deformation=args.initial_show_deformation,
            initial_deformation_scale=args.initial_deformation_scale,
        )
    else:
        # Static mode: load single result and save to file
        step, location = pick_step(headers, args.result, args.step, args.location)
        result_type, values = load_values(res_path, args.result, step, location)
        if not values:
            raise RuntimeError(
                f"Could not load values for result='{args.result}', step={step}, location='{location}'."
            )

        if location == "OnGaussPoints":
            print(f"Projecting {args.result} from OnGaussPoints to nodal values...")
            line_elements = []
            for _, _, elems in line_meshes:
                line_elements.extend(elems)
            values = project_element_to_nodes(values, surface_elements, line_elements if line_elements else None)

        out_path = Path(args.save)
        render_plot_static(
            all_nodes=all_nodes,
            triangles=triangles,
            values=values,
            result_name=args.result,
            step=step,
            result_type=result_type,
            component=args.component,
            tensor_component_mode=args.tensor_component,
            profile_nodes=profile_nodes,
            out_path=out_path,
            active_filter=active_filter,
            show_isolines=args.isolines,
            show_profile=args.profile,
            contour_levels=args.contour_levels,
            show_isoline_labels=args.isoline_labels,
            profile_path=profile_path,
        )
