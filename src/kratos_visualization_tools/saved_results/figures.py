"""Assemble configured series into figures and subplots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from .building import SeriesBuilder, _substitute
from .primitives import AxisOptions, DataSeries, make_single_plot, make_sub_plots


def _axis_options(spec: dict[str, Any], context: dict[str, Any]) -> AxisOptions:
    return AxisOptions(
        title=_substitute(spec.get("title"), context),
        xlabel=spec.get("xlabel"),
        ylabel=spec.get("ylabel"),
        xscale=spec.get("xscale"),
        yscale="log" if spec.get("log_y") else spec.get("yscale"),
        invert_yaxis=spec.get("invert_yaxis", False),
    )


def _collect(
    builder: SeriesBuilder,
    series_specs: list[dict],
    context: dict[str, Any],
    expand_over_stages: bool = False,
) -> list[DataSeries]:
    collection = []
    for series_spec in series_specs:
        stages = (
            context["stages"]
            if expand_over_stages and series_spec.get("expand_over_stages")
            else [context.get("stage")]
        )
        for stage in stages:
            series_context = {**context, "stage": stage}
            expanded_spec = series_spec
            if expand_over_stages and series_spec.get("expand_over_stages"):
                expanded_spec = {
                    **series_spec,
                    "label": _substitute(series_spec.get("label", ""), series_context),
                }
            series = builder.build(expanded_spec, series_context)
            if series is not None and series.data_points:
                collection.append(series)
    return collection


def _min_max_annotation(series_collection: list[DataSeries]) -> str:
    lines = []
    for series in series_collection:
        y_values = [point[1] for point in series.data_points]
        lines.append(f"Max. {series.label} = {max(y_values):12.3e}")
        lines.append(f"Min. {series.label} = {min(y_values):12.3e}")
    return "\n".join(lines)


def make_figure(
    figure_spec: dict[str, Any],
    builder: SeriesBuilder,
    output_dir: Path,
    default_stages: list[str],
):
    figure_type = figure_spec.get("type", "profiles_per_stage")
    stages = figure_spec.get("stages", default_stages)
    series_specs = figure_spec.get("series", [])
    base_context = {"stages": stages}

    if figure_type == "profiles_per_stage":
        collections, options = [], []
        for stage in stages:
            context = {**base_context, "stage": stage, "node": None}
            collection = _collect(builder, series_specs, context)
            if not collection:
                continue
            collections.append(collection)
            options.append(_axis_options({**figure_spec, "title": stage}, context))
        _emit_sub_plots(figure_spec, collections, options, output_dir)

    elif figure_type == "histories_per_node":
        node_ids = builder.node_ids(figure_spec)
        collections, options = [], []
        for node_id in node_ids:
            context = {**base_context, "node": node_id, "stage": None}
            collection = _collect(builder, series_specs, context)
            if not collection:
                continue
            collections.append(collection)
            options.append(
                _axis_options({**figure_spec, "title": f"Node {node_id}"}, context)
            )
        _emit_sub_plots(figure_spec, collections, options, output_dir)

    elif figure_type == "custom_subplots":
        collections, options = [], []
        for subplot_spec in figure_spec["subplots"]:
            context = {**base_context, "stage": None, "node": None}
            collections.append(_collect(builder, subplot_spec["series"], context))
            options.append(_axis_options(subplot_spec, context))
        _emit_sub_plots(figure_spec, collections, options, output_dir)

    elif figure_type == "single":
        target_stages = stages if figure_spec.get("per_stage_files") else [None]
        for stage in target_stages:
            context = {**base_context, "stage": stage, "node": None}
            collection = _collect(
                builder,
                series_specs,
                context,
                expand_over_stages=not figure_spec.get("per_stage_files", False),
            )
            if not collection:
                print(f"Warning: no data for figure '{figure_spec['name']}'; skipped.")
                continue
            annotation = (
                _min_max_annotation(collection)
                if figure_spec.get("annotate_min_max")
                else None
            )
            make_single_plot(
                collection,
                output_dir / _substitute(figure_spec["name"], context),
                _axis_options(figure_spec, context),
                annotation,
            )

    else:
        raise ValueError(f"Unknown figure type '{figure_type}'")


def _emit_sub_plots(
    figure_spec: dict[str, Any],
    collections: list[list[DataSeries]],
    options: list[AxisOptions],
    output_dir: Path,
):
    if not collections:
        print(f"Warning: no data for figure '{figure_spec['name']}'; skipped.")
        return
    make_sub_plots(
        collections,
        output_dir / figure_spec["name"],
        options,
        max_plots_per_row=figure_spec.get("max_plots_per_row", 5),
        figsize=tuple(figure_spec.get("figsize", (20, 6))),
    )
