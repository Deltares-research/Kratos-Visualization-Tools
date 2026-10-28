import json
from types import SimpleNamespace

import numpy as np
import pytest

import visualize_gid_results as viz

from conftest import (
    close_viewer_figures,
    write_mesh,
    write_results,
    write_second_mesh,
    write_second_results,
)


def test_comparison_reports_common_and_unique_variables():
    first = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("FIRST_ONLY", 1, "Scalar", "OnNodes", None, 2),
    ]
    second = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("SECOND_ONLY", 1, "Scalar", "OnNodes", None, 2),
    ]

    assert viz.compare_result_names(first, second) == (
        ["PRESSURE"],
        ["FIRST_ONLY"],
        ["SECOND_ONLY"],
    )


def test_comparison_viewer_reset_and_adjust_scale():
    viewer = viz.ComparisonViewer.__new__(viz.ComparisonViewer)
    viewer.current_style = "background+isolines"
    viewer.fields = [
        {"scalar": np.array([1.0, np.nan])},
        {"scalar": np.array([-2.0, 4.0])},
    ]
    viewer.difference_field = {"scalar": np.array([-2.0, 5.0])}
    viewer.requested_vmin = 10.0
    viewer.requested_vmax = 20.0
    viewer.vmin = 10.0
    viewer.vmax = 20.0

    viewer._reset_scale()
    assert viewer.requested_vmin is None
    assert viewer.requested_vmax is None
    assert (viewer.vmin, viewer.vmax) == (-2.0, 4.0)

    viewer._adjust_scale(0.5)
    assert (viewer.requested_vmin, viewer.requested_vmax) == (-0.5, 2.5)
    assert (viewer.vmin, viewer.vmax) == (-0.5, 2.5)

    viewer.current_style = "difference"
    viewer._reset_scale()
    assert (viewer.vmin, viewer.vmax) == (-5.0, 5.0)


def test_weighted_coordinate_inspection_uses_nearby_values(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    solution = viz.load_solution_data("model", "model", base_dir=tmp_path)
    values = {
        1: np.array([10.0]),
        2: np.array([20.0]),
        3: np.array([30.0]),
    }

    nearest = viz.ComparisonViewer._value_at_coordinate(solution, values, 0.5, 0.5, "nearest")
    weighted = viz.ComparisonViewer._value_at_coordinate(solution, values, 0.5, 0.5, "idw")

    assert float(nearest[3][0]) == 10.0
    assert float(weighted[3][0]) == 20.0


def test_comparison_loads_different_meshes_and_saves_shared_plot(tmp_path):
    first_mesh = tmp_path / "first.post.msh"
    first_results = tmp_path / "first.post.res"
    second_mesh = tmp_path / "second.post.msh"
    second_results = tmp_path / "second.post.res"
    write_mesh(first_mesh)
    write_results(first_results)
    write_second_mesh(second_mesh)
    write_second_results(second_results)

    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)
    common, first_only, second_only = viz.compare_result_names(first.headers, second.headers)
    assert common == ["DISPLACEMENT", "PRESSURE"]
    assert first_only == ["STRESS"]
    assert second_only == ["TEMPERATURE"]

    output_path = tmp_path / "comparison.png"
    viz.render_comparison(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
        out_path=output_path,
    )

    assert output_path.is_file()

    difference_path = tmp_path / "difference.png"
    viz.render_comparison(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="difference",
        vmin=None,
        vmax=None,
        out_path=difference_path,
    )
    assert difference_path.is_file()


def test_comparison_viewer_update_plot_supports_all_styles(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)

    viewers = []
    try:
        for style in viz.ComparisonViewer.styles:
            viewer = viz.ComparisonViewer(
                first,
                second,
                initial_result="PRESSURE",
                initial_step="latest",
                initial_location=None,
                component="magnitude",
                tensor_component_mode="trace",
                profile_x=0.0,
                style=style,
                vmin=None,
                vmax=None,
            )
            viewer.update_plot()
            assert viewer.current_style == style
            assert viewer.fig is not None
            viewers.append(viewer)
    finally:
        for viewer in viewers:
            close_viewer_figures(viewer)


def test_comparison_viewer_shows_comparison_error_on_all_axes(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)

    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
        show_profile=True,
    )
    try:
        viewer._show_comparison_error(RuntimeError("comparison failed"))

        assert any(
            text.get_text() == "Error: comparison failed"
            for text in viewer.ax0.texts
        )
        assert not viewer.ax0.axison
        assert not viewer.ax1.axison
        assert not viewer.ax2.axison
    finally:
        close_viewer_figures(viewer)


def load_comparison_solutions(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    return (
        viz.load_solution_data("first", "first", base_dir=tmp_path),
        viz.load_solution_data("second", "second", base_dir=tmp_path),
    )


def test_comparison_popup_table_reports_both_nodes_and_missing_variables(tmp_path):
    first, second = load_comparison_solutions(tmp_path)

    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
    )
    try:
        assert viewer.ax1 is None
        columns, rows = viewer._build_comparison_table(0.0, 0.0)
        viewer.inspection_method = "idw"
        idw_columns, idw_rows = viewer._build_comparison_table(0.5, 0.5)
        assert "IDW at (0.5, 0.5)" in idw_columns[1]
        assert "nearest node 1" in idw_columns[1]
        assert "IDW at (0.5, 0.5)" in idw_columns[2]
        assert "nearest node 101" in idw_columns[2]
        assert len(idw_rows) == len(rows)
        viewer._show_comparison_popup(0.5, 0.5)
        assert viewer.info_fig is not None
        viewer._show_comparison_popup(0.75, 0.75)
    finally:
        close_viewer_figures(viewer)

    assert "node 1" in columns[1]
    assert "node 101" in columns[2]
    rows_by_name = {row[0]: row for row in rows}
    assert "10" in rows_by_name["PRESSURE"][1]
    assert "20" in rows_by_name["PRESSURE"][2]
    assert rows_by_name["STRESS"][2] == "-"
    assert rows_by_name["TEMPERATURE"][1] == "-"


def test_comparison_click_and_coordinate_input_open_popup(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
    )
    try:
        popup_coordinates = []
        viewer._show_comparison_popup = lambda x, y: popup_coordinates.append((x, y))
        viewer.on_click(SimpleNamespace(
            inaxes=viewer.ax0, button=1, xdata=0.25, ydata=0.5
        ))
        viewer.on_click(SimpleNamespace(
            inaxes=viewer.ax0, button=3, xdata=0.25, ydata=0.5
        ))
        viewer.on_click(SimpleNamespace(
            inaxes=None, button=1, xdata=0.25, ydata=0.5
        ))
        viewer.on_click(SimpleNamespace(
            inaxes=viewer.ax0, button=1, xdata=None, ydata=0.5
        ))
        assert popup_coordinates == [(0.25, 0.5)]

        viewer._open_coordinate_input_popup()
        viewer.coord_textbox._observers.process("submit", "invalid")
        assert "exactly two numbers" in viewer.coord_status_artist.get_text()
        viewer.coord_textbox._observers.process("submit", "x, y")
        assert "Invalid numeric format" in viewer.coord_status_artist.get_text()
        viewer.coord_textbox._observers.process("submit", "0.75, 1.25")
        assert popup_coordinates == [(0.25, 0.5), (0.75, 1.25)]
    finally:
        close_viewer_figures(viewer)


def test_comparison_mesh_toggle_updates_lines(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
    )
    try:
        viewer.show_mesh = True
        viewer.update_plot()
        mesh_line_styles = {line.get_linestyle() for line in viewer.ax0.lines}
        assert "-" in mesh_line_styles
        assert "--" in mesh_line_styles
    finally:
        close_viewer_figures(viewer)


def make_comparison_viewer(first, second, **overrides):
    options = {
        "initial_result": "PRESSURE",
        "initial_step": "latest",
        "initial_location": None,
        "component": "magnitude",
        "tensor_component_mode": "trace",
        "profile_x": 0.0,
        "style": "background+isolines",
        "vmin": None,
        "vmax": None,
    }
    options.update(overrides)
    return viz.ComparisonViewer(first, second, **options)


def test_comparison_profile_and_menu_dragging(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        colormap="cividis",
    )
    try:
        assert viewer.ax1 is not None
        assert viewer.ax2 is not None
        assert viewer.current_colormap == "cividis"
        profile_line_count = len(viewer.ax1.lines)
        profile_data = viewer._draw_profile(
            viewer.solutions[0], viewer.fields[0], "green"
        )
        assert profile_data is not None
        assert len(viewer.ax1.lines) == profile_line_count + 1

        menu_artist = viewer.menu_artists[0]
        viewer._on_menu_pick(SimpleNamespace(
            artist=menu_artist,
            mouseevent=SimpleNamespace(x=100, y=100),
        ))
        viewer._on_menu_motion(SimpleNamespace(x=150, y=140))
        viewer._on_menu_release(SimpleNamespace())
        assert viewer.menu_drag_artist is None
    finally:
        close_viewer_figures(viewer)


def test_comparison_viewer_renders_profile_path(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        profile_path=[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
    )
    try:
        assert viewer.ax1 is not None
        assert viewer.ax1.get_ylabel() == "distance"
        assert viewer._get_profile_data(viewer.solutions[0], viewer.fields[0]) is not None
    finally:
        close_viewer_figures(viewer)


def test_comparison_settings_and_scale_input(tmp_path, monkeypatch):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        colormap="cividis",
    )
    try:
        picture_path = viewer.save_current_plot(tmp_path / "hotkey_picture.png")
        assert picture_path.is_file()
        settings_path = viewer.save_current_settings(tmp_path / "comparison_settings.json")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["mode"] == "comparison"
        assert len(settings["solutions"]) == 2

        viewer._open_scale_input_popup()
        viewer.scale_textbox._observers.process("submit", "invalid")
        assert "exactly two numbers" in viewer.scale_status_artist.get_text()
        viewer.scale_textbox._observers.process("submit", "10, 5")
        assert "vmin < vmax" in viewer.scale_status_artist.get_text()
        viewer.scale_textbox._observers.process("submit", "-10, 50")
        assert (viewer.vmin, viewer.vmax) == (-10.0, 50.0)
        monkeypatch.chdir(tmp_path)
        assert viewer.save_current_plot().is_file()
    finally:
        close_viewer_figures(viewer)


def test_comparison_keyboard_dispatch(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        colormap="cividis",
    )
    try:
        calls = []
        viewer.update_plot = lambda: calls.append("update")
        viewer.save_current_plot = lambda: calls.append("save")
        viewer._open_coordinate_input_popup = lambda: calls.append("coordinate")
        viewer._open_scale_input_popup = lambda: calls.append("scale")
        for key in ("n", "p", "o", "c", "l", "L", "m", "b", "a", "w", "j", "k", "[", "]", "i", "S", "v"):
            viewer.on_key(SimpleNamespace(key=key))
        viewer.on_key(SimpleNamespace(key="s"))
        assert "save" in calls
        assert "coordinate" in calls
        assert "scale" in calls
        assert calls.count("update") == 14
        viewer.on_key(SimpleNamespace(key="q"))
    finally:
        close_viewer_figures(viewer)


def test_comparison_viewer_rejects_invalid_options(tmp_path):
    first, second = load_comparison_solutions(tmp_path)

    with pytest.raises(ValueError, match="Inspection method"):
        make_comparison_viewer(
            first,
            second,
            inspection_method="invalid",
        )

    with pytest.raises(ValueError, match="Unsupported comparison colormap"):
        make_comparison_viewer(
            first,
            second,
            colormap="invalid",
        )
