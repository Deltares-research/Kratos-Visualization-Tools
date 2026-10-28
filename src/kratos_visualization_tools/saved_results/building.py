"""Build styled series from saved JSON and tabular data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from .data import Mesh, read_table
from .primitives import DataSeries


VALUE_SOURCES = ("value", "node_x", "node_y", "node_z", "node_id", "time")


def _substitute(template: Any, context: dict[str, Any]) -> Any:
    if not isinstance(template, str):
        return template
    result = template
    for key, value in context.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _transform(value: float, spec: dict[str, Any]) -> float:
    return value * float(spec.get("scale", 1.0)) + float(spec.get("offset", 0.0))


def _coordinate(mesh: Mesh, node_id: int, source: str) -> float:
    if source == "node_id":
        return float(node_id)
    if node_id not in mesh.coordinates:
        raise KeyError(
            f"Node {node_id} has no coordinates; provide a mesh file or node coordinates"
        )
    return mesh.coordinates[node_id]["node_x node_y node_z".split().index(source)]


def _nodal_value(json_data: dict, node_id: int, spec: dict[str, Any]) -> float | None:
    node_entry = json_data.get(f"NODE_{node_id}")
    if node_entry is None:
        return None
    values = node_entry.get(spec["variable"])
    if not values:
        return None
    value = values[int(spec.get("time_index", 0))]
    component = spec.get("component")
    if component is not None:
        if not isinstance(value, list) or component >= len(value):
            return None
        value = value[component]
    return _transform(float(value), spec)


def _sorted_points(
    points: list[tuple[float, float]], sort_by: str | None
) -> list[tuple[float, float]]:
    if sort_by == "x":
        return sorted(points, key=lambda point: point[0])
    if sort_by == "y":
        return sorted(points, key=lambda point: point[1])
    return points


def _series_style(spec: dict[str, Any], points: list[tuple[float, float]]) -> DataSeries:
    return DataSeries(
        data_points=points,
        label=spec.get("label", ""),
        line_style=spec.get("line_style", "-"),
        marker=spec.get("marker"),
        color=spec.get("color"),
    )


class SeriesBuilder:
    def __init__(self, data_dir: Path, mesh: Mesh, node_sets: dict[str, list[int]]):
        self.data_dir = data_dir
        self.mesh = mesh
        self.node_sets = node_sets

    def node_ids(self, spec: dict[str, Any]) -> list[int]:
        if "node_ids" in spec:
            return [int(node_id) for node_id in spec["node_ids"]]
        if "node_set" in spec:
            if spec["node_set"] not in self.node_sets:
                raise KeyError(f"Unknown node set '{spec['node_set']}'")
            return self.node_sets[spec["node_set"]]
        raise KeyError("A series on Kratos output needs 'node_set' or 'node_ids'")

    def build(
        self, spec: dict[str, Any], context: dict[str, Any]
    ) -> DataSeries | None:
        source = spec.get("source", "kratos_json")
        if source == "kratos_json":
            if context.get("node") is not None and spec.get("over", "stages") == "stages":
                return self._build_history(spec, context)
            return self._build_profile(spec, context)
        if source == "csv":
            return self._build_from_table(spec, context)
        raise ValueError(f"Unknown series source '{source}'")

    def _open_json(self, spec: dict[str, Any], context: dict[str, Any]) -> dict | None:
        file_path = self.data_dir / _substitute(spec["file"], context)
        if not file_path.exists():
            print(f"Warning: JSON file {file_path} does not exist; skipping series.")
            return None
        with open(file_path, "r") as json_file:
            return json.load(json_file)

    def _build_profile(
        self, spec: dict[str, Any], context: dict[str, Any]
    ) -> DataSeries | None:
        json_data = self._open_json(spec, context)
        if json_data is None:
            return None

        x_source = spec.get("x", "value")
        y_source = spec.get("y", "node_y")
        node_ids = (
            [int(context["node"])] if context.get("node") is not None else self.node_ids(spec)
        )

        points = []
        for node_id in node_ids:
            coordinates = {}
            for axis, source in (("x", x_source), ("y", y_source)):
                if source == "value":
                    value = _nodal_value(json_data, node_id, spec)
                elif source == "time":
                    value = float(json_data["TIME"][int(spec.get("time_index", 0))])
                else:
                    value = _coordinate(self.mesh, node_id, source)
                coordinates[axis] = value
            if coordinates["x"] is None or coordinates["y"] is None:
                continue
            points.append((coordinates["x"], coordinates["y"]))

        points = _sorted_points(points, spec.get("sort_by", "y"))
        return _series_style(spec, points)

    def _build_history(
        self, spec: dict[str, Any], context: dict[str, Any]
    ) -> DataSeries | None:
        node_id = int(context["node"])
        points = []
        for stage in spec.get("stages") or context["stages"]:
            stage_context = {**context, "stage": stage}
            json_data = self._open_json(spec, stage_context)
            if json_data is None:
                continue
            value = _nodal_value(json_data, node_id, spec)
            if value is None:
                continue
            # Nodes that are not active yet report an exact zero; skip those.
            if spec.get("skip_zero_values", True) and value == 0.0:
                continue
            time = float(json_data["TIME"][int(spec.get("time_index", 0))])
            points.append((time, value))

        return _series_style(spec, _sorted_points(points, spec.get("sort_by", "x")))

    def _build_from_table(
        self, spec: dict[str, Any], context: dict[str, Any]
    ) -> DataSeries | None:
        file_path = self.data_dir / _substitute(spec["file"], context)
        if not file_path.exists():
            print(f"Warning: data file {file_path} does not exist; skipping series.")
            return None

        table = read_table(file_path, spec.get("comment_prefix", "#"))
        x_key = _substitute(spec.get("x_column", 0), context)
        y_key = _substitute(spec.get("y_column", 1), context)
        if not table.has_column(x_key) or not table.has_column(y_key):
            print(
                f"Warning: columns '{x_key}'/'{y_key}' not found in {file_path}; skipping series."
            )
            return None

        row_filter = {
            column: str(_substitute(value, context))
            for column, value in spec.get("filter", {}).items()
        }
        for column in row_filter:
            if not table.has_column(column):
                print(
                    f"Warning: filter column '{column}' not found in {file_path}; skipping series."
                )
                return None
        filter_columns = {column: table.column(column) for column in row_filter}
        kept_rows = [
            index
            for index in range(len(table.rows))
            if all(filter_columns[column][index] == value for column, value in row_filter.items())
        ]
        x_cells = table.column(x_key)
        y_cells = table.column(y_key)

        missing_value = spec.get("missing_value", "N/A")
        x_spec = {"scale": spec.get("x_scale", 1.0), "offset": spec.get("x_offset", 0.0)}
        y_spec = {"scale": spec.get("y_scale", 1.0), "offset": spec.get("y_offset", 0.0)}

        points = []
        for index in kept_rows:
            x_cell, y_cell = x_cells[index], y_cells[index]
            if missing_value in (x_cell, y_cell):
                continue
            points.append(
                (_transform(float(x_cell), x_spec), _transform(float(y_cell), y_spec))
            )

        return _series_style(spec, _sorted_points(points, spec.get("sort_by")))
