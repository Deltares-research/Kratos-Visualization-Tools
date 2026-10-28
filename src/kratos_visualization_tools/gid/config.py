"""GiD visualization settings and profile-path configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def parse_profile_path(value) -> Optional[List[Tuple[float, float]]]:
    if value is None or value == "":
        return None

    raw_points = value.split(";") if isinstance(value, str) else value
    points = []
    try:
        for raw_point in raw_points:
            if isinstance(raw_point, str):
                coordinates = raw_point.split(",")
            else:
                coordinates = raw_point
            if len(coordinates) != 2:
                raise ValueError
            points.append((float(coordinates[0]), float(coordinates[1])))
    except (TypeError, ValueError):
        raise ValueError("Profile path must contain points formatted as x,y;x,y;...")

    if len(points) < 2:
        raise ValueError("Profile path must contain at least two x,y points.")
    return points


def load_visualization_settings(settings_path: Path) -> Dict[str, object]:
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid settings JSON in {settings_path}: {exc}") from exc

    if not isinstance(settings, dict):
        raise ValueError(f"Settings file must contain a JSON object: {settings_path}")
    mode = settings.get("mode")
    if mode not in ("single", "comparison"):
        raise ValueError("Settings mode must be 'single' or 'comparison'.")
    return settings


def apply_visualization_settings(args, settings: Dict[str, object]):
    for argument, setting in (
        ("result", "result"),
        ("step", "step"),
        ("location", "location"),
        ("component", "component"),
        ("tensor_component", "tensor_component"),
        ("profile_x", "profile_x"),
        ("profile_path", "profile_path"),
        ("x_tol", "x_tol"),
        ("active", "active_submodelpart"),
        ("isolines", "show_isolines"),
        ("isoline_labels", "show_isoline_labels"),
        ("profile", "show_profile"),
        ("colormap", "colormap"),
        ("comparison_style", "style"),
        ("vmin", "vmin"),
        ("vmax", "vmax"),
        ("inspection_method", "inspection_method"),
    ):
        if setting in settings and settings[setting] is not None:
            setattr(args, argument, settings[setting])

    if "show_domain_edge" in settings:
        args.no_domain_edge = not bool(settings["show_domain_edge"])
    args.initial_show_gradient = bool(settings.get("show_gradient", False))
    args.initial_show_mesh = bool(settings.get("show_mesh", False))
    args.initial_colormap = settings.get("colormap")
    args.initial_show_deformation = bool(settings.get("show_deformation", False))
    args.initial_deformation_scale = float(settings.get("deformation_scale", 1.0))
