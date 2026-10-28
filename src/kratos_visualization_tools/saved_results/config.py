"""Expand plot templates, discover stages, and execute configurations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from .building import SeriesBuilder
from .data import Mesh, read_mdpa
from .figures import make_figure
from .primitives import PendingFigure
from . import primitives


CONFIG_DOC = """
{
  "output_dir": "plots",                  // relative to the config file
  "data_dir": ".",                        // where the JSON/CSV data lives
  "mesh": {"mdpa_file": "common/model.mdpa"},
  "node_sets": {                          // named groups of nodes
    "wall": {"sub_model_part": "Sheet_Pile_Wall"},
    "probes": {"ids": [12, 34, 56]}
  },
    "stages": "auto",                       // or an explicit list of stage names
    "figure_defaults": {"type": "profiles_per_stage", "ylabel": "$y$ [m]"},
    "figure_templates": {
        "wall_profile": {"type": "profiles_per_stage", "ylabel": "$y$ [m]"}
    },
    "series_templates": {
        "kratos_wall": {"file": "{stage}__output_wall.json", "node_set": "wall"}
    },
  "figures": [
    {
            "template": "wall_profile",
            "name": "moments.svg",
            "series": [{"template": "kratos_wall", "variable": "BENDING_MOMENT"}]
    }
  ]
}
"""


def run(
    config_file_path: Path,
    output_dir_override: Path | None = None,
    show: bool = False,
) -> list[PendingFigure]:
    primitives._pending_figures = [] if show else None
    try:
        _make_figures(config_file_path, output_dir_override)
        return primitives._pending_figures or []
    finally:
        primitives._pending_figures = None


def _expand_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    series_templates = config.get("series_templates", {})
    figure_templates = config.get("figure_templates", {})

    def expand_series(series_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded = []
        for series_spec in series_specs:
            template_name = series_spec.get("template")
            if template_name is None:
                expanded.append(series_spec)
                continue
            try:
                template = series_templates[template_name]
            except KeyError as error:
                raise KeyError(f"Unknown series template '{template_name}'") from error
            overrides = {
                key: value for key, value in series_spec.items() if key != "template"
            }
            expanded.append({**template, **overrides})
        return expanded

    figures = []
    for figure_spec in config["figures"]:
        template_name = figure_spec.get("template")
        figure_template = {}
        if template_name is not None:
            try:
                figure_template = figure_templates[template_name]
            except KeyError as error:
                raise KeyError(f"Unknown figure template '{template_name}'") from error
        figure_overrides = {
            key: value for key, value in figure_spec.items() if key != "template"
        }
        figure = {
            **config.get("figure_defaults", {}),
            **figure_template,
            **figure_overrides,
        }
        if "series" in figure:
            figure["series"] = expand_series(figure["series"])
        if "subplots" in figure:
            figure["subplots"] = [
                {
                    **subplot,
                    "series": expand_series(subplot["series"]),
                }
                if "series" in subplot
                else subplot
                for subplot in figure["subplots"]
            ]
        figures.append(figure)
    return figures


def _natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


def _discover_stages(data_dir: Path, figures: list[dict[str, Any]]) -> list[str]:
    stages = set()
    for figure in figures:
        series_specs = list(figure.get("series", []))
        for subplot in figure.get("subplots", []):
            series_specs.extend(subplot.get("series", []))

        for series_spec in series_specs:
            file_pattern = series_spec.get("file")
            if not isinstance(file_pattern, str) or file_pattern.count("{stage}") != 1:
                continue

            normalized_pattern = file_pattern.replace("\\", "/")
            glob_pattern = normalized_pattern.replace("{stage}", "*")
            stage_pattern = re.compile(
                re.escape(normalized_pattern).replace(re.escape("{stage}"), r"([^/]+)")
            )
            for file_path in data_dir.glob(glob_pattern):
                if not file_path.is_file():
                    continue
                relative_path = file_path.relative_to(data_dir).as_posix()
                match = stage_pattern.fullmatch(relative_path)
                if match:
                    stages.add(match.group(1))

    if not stages:
        raise ValueError(
            "Could not auto-discover stages: no files matched a series file pattern "
            "containing '{stage}'."
        )
    return sorted(stages, key=_natural_sort_key)


def _make_figures(config_file_path: Path, output_dir_override: Path | None):
    with open(config_file_path, "r") as config_file:
        config = json.load(config_file)

    config_dir = config_file_path.parent.resolve()
    data_dir = (config_dir / config.get("data_dir", ".")).resolve()
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override
        else (config_dir / config.get("output_dir", ".")).resolve()
    )

    mesh = Mesh()
    mesh_spec = config.get("mesh", {})
    if "mdpa_file" in mesh_spec:
        mesh = read_mdpa(data_dir / mesh_spec["mdpa_file"])
    for node_id, coordinates in mesh_spec.get("node_coordinates", {}).items():
        mesh.coordinates[int(node_id)] = tuple(
            list(coordinates) + [0.0] * (3 - len(coordinates))
        )

    node_sets = {}
    for name, node_set_spec in config.get("node_sets", {}).items():
        if "ids" in node_set_spec:
            node_sets[name] = [int(node_id) for node_id in node_set_spec["ids"]]
        else:
            node_sets[name] = mesh.node_ids(node_set_spec["sub_model_part"])

    builder = SeriesBuilder(data_dir, mesh, node_sets)
    figures = _expand_config(config)
    stages = config.get("stages", [])
    if stages == "auto":
        stages = _discover_stages(data_dir, figures)
    for figure_spec in figures:
        make_figure(figure_spec, builder, output_dir, stages)
