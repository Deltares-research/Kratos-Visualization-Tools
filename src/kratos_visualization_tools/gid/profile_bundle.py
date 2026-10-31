"""Portable profile data and saved-results plotting configurations."""

import csv
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from .results import resolve_mesh_result_paths


def save_profile_bundle(viewer, parent, *, include_difference=False):
    """Export exactly the current profile values without reloading GiD results."""
    comparison = hasattr(viewer, "solutions")
    if include_difference and not comparison:
        raise ValueError("A profile difference requires two solutions.")
    profiles = []
    sources = []
    if comparison:
        for index, (solution, field) in enumerate(zip(viewer.solutions, viewer.fields)):
            profile = viewer._get_profile_data(solution, field)
            if profile is None:
                raise ValueError(f"No profile data for {solution.label}. Choose a usable profile.")
            profiles.append((f"solution_{index + 1}", solution.label, profile))
            sources.append({
                "label": solution.label, "mesh": str(solution.msh_path),
                "results": str(solution.res_path), "step": field["step"],
                "location": field["location"], "result_type": field["result_type"],
            })
        if include_difference:
            difference = viewer._difference_profile(profiles[0][2], profiles[1][2])
            if difference is None:
                raise ValueError("No overlapping profile data for the difference.")
            profiles.append(("difference", "Solution 2 - solution 1", difference))
    else:
        if viewer.profile_y_data is None or viewer.profile_values is None:
            raise ValueError("No profile data. Enable Profile and choose a usable profile.")
        profiles.append(("solution_1", viewer.res_path.stem,
                         (viewer.profile_y_data, viewer.profile_values)))
        mesh, results = resolve_mesh_result_paths(str(viewer.res_path))
        sources.append({
            "label": viewer.res_path.stem, "mesh": str(mesh), "results": str(results),
            "step": viewer.get_current_step(), "location": viewer.current_location,
        })

    rows = []
    for key, label, (coordinates, values) in profiles:
        coordinates = np.asarray(coordinates, dtype=float)
        values = np.asarray(values, dtype=float)
        if coordinates.ndim != 1 or coordinates.shape != values.shape:
            raise ValueError(f"Invalid profile array shapes for {label}.")
        finite = np.isfinite(coordinates) & np.isfinite(values)
        if not np.any(finite):
            raise ValueError(f"No finite profile data for {label}.")
        rows.extend((key, float(coordinate), float(value))
                    for coordinate, value in zip(coordinates[finite], values[finite]))

    component = viewer.current_component
    tensor = viewer.current_tensor_component
    gradient = getattr(viewer, "show_gradient", False)
    title = f"{viewer.current_result} profile"
    xlabel = f"{viewer.current_result} ({component}; tensor: {tensor})"
    if gradient:
        xlabel += " gradient magnitude"
    metadata = {
        "mode": "comparison" if comparison else "single",
        "sources": sources, "result": viewer.current_result,
        "requested_step": viewer.current_step_str,
        "component": component, "tensor_component": tensor,
        "gradient": gradient, "profile_x": viewer.profile_x,
        "profile_path": viewer.profile_path,
        "difference": "solution_2 - solution_1" if include_difference else None,
        "coordinate": "distance_along_profile" if viewer.profile_path else "y",
        "sampling": "current GiD viewer profile",
        "difference_interpolation": "linear on overlapping solution 2 coordinates"
        if include_difference else None,
    }
    series = [{
        "source": "csv", "file": "profile.csv", "x_column": "value",
        "y_column": "coordinate", "filter": {"series": key}, "label": label,
        "sort_by": "y",
    } for key, label, _ in profiles if key != "difference"]
    figures = [{
        "name": "profile.svg", "type": "single", "title": title,
        "xlabel": xlabel, "ylabel": "Distance along profile" if viewer.profile_path else "y",
        "series": series,
    }]
    if include_difference:
        figures.append({
            "name": "difference.svg", "type": "single", "title": f"{title} difference",
            "xlabel": f"{viewer.current_result}: solution 2 - solution 1",
            "ylabel": figures[0]["ylabel"],
            "series": [{
                "source": "csv", "file": "profile.csv", "x_column": "value",
                "y_column": "coordinate", "filter": {"series": "difference"},
                "label": "Solution 2 - solution 1", "sort_by": "y",
            }],
        })
    config = {"data_dir": ".", "output_dir": "plots", "metadata": metadata,
              "figures": figures}
    folder = Path(parent) / (
        f"profile_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}")
    folder.mkdir()
    config_path = folder / "plot_config.json"
    try:
        with (folder / "profile.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["series", "coordinate", "value"])
            writer.writerows(rows)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except (OSError, ValueError, TypeError):
        config_path.unlink(missing_ok=True)
        (folder / "profile.csv").unlink(missing_ok=True)
        folder.rmdir()
        raise
    return config_path
