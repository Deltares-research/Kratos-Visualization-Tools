import numpy as np
import pytest

import visualize_gid_results as viz

from conftest import (
    write_mesh,
    write_results,
)


def test_scan_and_load_results_support_steps_and_result_types(tmp_path):
    result_path = tmp_path / "model.post.res"
    write_results(result_path)

    headers = viz.scan_result_headers(result_path)
    assert [(header.name, header.step, header.location) for header in headers] == [
        ("PRESSURE", 1, "OnNodes"),
        ("PRESSURE", 2, "OnNodes"),
        ("DISPLACEMENT", 2, "OnNodes"),
        ("STRESS", 2, "OnGaussPoints"),
    ]

    result_type, values = viz.load_values(result_path, "DISPLACEMENT", 2, "OnNodes")
    assert result_type == "Vector"
    np.testing.assert_allclose(values[1], [3.0, 4.0])


def test_pick_step_prefers_nodes_and_validates_requests():
    headers = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnGaussPoints", None, 1),
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnGaussPoints", None, 2),
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnNodes", None, 3),
    ]

    assert viz.pick_step(headers, "PRESSURE", "latest", None) == (2, "OnNodes")
    assert viz.pick_step(headers, "PRESSURE", "1.0", "OnGaussPoints") == (1, "OnGaussPoints")

    with pytest.raises(ValueError, match="Invalid step"):
        viz.pick_step(headers, "PRESSURE", "not-a-step", "OnNodes")
    with pytest.raises(ValueError, match="Step 4 not found"):
        viz.pick_step(headers, "PRESSURE", "4", "OnNodes")


def test_get_available_steps_returns_sorted_unique_steps_for_current_result():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.current_result = "PRESSURE"
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("OTHER", 9, "Scalar", "OnNodes", None, 2),
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 3),
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 4),
    ]

    assert viewer.get_available_steps() == [1, 3]


def test_select_step_and_location_prefers_nodes_then_gauss_points():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 2),
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnGaussPoints", "GP", 3),
        viz.ResultHeader("PRESSURE", 4, "Scalar", "OnGaussPoints", "GP", 4),
    ]

    assert viewer._select_step_and_location_for_result("PRESSURE", 1) == (1, "OnNodes")
    assert viewer._select_step_and_location_for_result("PRESSURE", 2) == (2, "OnGaussPoints")
    assert viewer._select_step_and_location_for_result("PRESSURE", 8) == (3, "OnNodes")


def test_select_step_and_location_falls_back_to_latest_gauss_point_or_errors():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnGaussPoints", "GP", 1),
        viz.ResultHeader("PRESSURE", 5, "Scalar", "OnGaussPoints", "GP", 2),
    ]

    assert viewer._select_step_and_location_for_result("PRESSURE", 8) == (5, "OnGaussPoints")
    with pytest.raises(ValueError, match="No available steps for result 'MISSING'"):
        viewer._select_step_and_location_for_result("MISSING", 1)


def test_format_result_line_handles_empty_scalar_vector_and_matrix_values():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)

    assert viewer._format_result_line("EMPTY", 1, "OnNodes", "Scalar", np.array([])) == (
        "EMPTY: <empty> [step 1, OnNodes]"
    )
    assert viewer._format_result_line("PRESSURE", 2, "OnNodes", "Scalar", np.array([12.3456789])) == (
        "PRESSURE: 12.3457 [step 2, OnNodes]"
    )

    vector_line = viewer._format_result_line(
        "DISPLACEMENT", 3, "OnNodes", "Vector", np.array([3.0, 4.0])
    )
    assert vector_line == (
        "DISPLACEMENT: x=3, y=4, z=nan, |v|=5 [step 3, OnNodes]"
    )

    matrix_line = viewer._format_result_line(
        "STRESS", 4, "OnGaussPoints", "Matrix", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )
    assert matrix_line == (
        "STRESS: xx=1, xy=4, yx=4, yy=2, trace=6, |T|=9.53939 "
        "[step 4, OnGaussPoints]"
    )


def test_build_node_report_formats_values_missing_nodes_and_errors():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.result_names = ["PRESSURE", "DISPLACEMENT", "STRESS", "MISSING", "BROKEN"]
    viewer.headers = [viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 1)]
    viewer.get_current_step = lambda: 2
    viewer._select_step_and_location_for_result = lambda name, step: (step, "OnNodes")

    result_data = {
        "PRESSURE": ("Scalar", {7: np.array([12.5])}),
        "DISPLACEMENT": ("Vector", {7: np.array([3.0, 4.0])}),
        "STRESS": ("Matrix", {7: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}),
        "MISSING": ("Scalar", {}),
    }

    def load_cached_values(name, step, location):
        if name == "BROKEN":
            raise RuntimeError("load failed")
        return result_data[name]

    viewer._load_cached_values = load_cached_values
    report = viewer._build_node_report(1.25, 2.5, 7, 1.0, 2.0, 0.25)

    assert "Mouse click: x=1.25, y=2.5" in report
    assert "Nearest node: id=7, x=1, y=2" in report
    assert "PRESSURE: 12.5 [step 2, OnNodes]" in report
    assert "DISPLACEMENT: x=3, y=4, z=nan, |v|=5 [step 2, OnNodes]" in report
    assert "STRESS: xx=1, xy=4, yx=4, yy=2, trace=6, |T|=9.53939 [step 2, OnNodes]" in report
    assert "MISSING: <no value> [step 2, OnNodes]" in report
    assert "BROKEN: <error: load failed>" in report


def test_build_node_report_uses_latest_header_step_when_current_step_fails():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.result_names = ["PRESSURE"]
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("PRESSURE", 5, "Scalar", "OnNodes", None, 2),
    ]
    viewer.get_current_step = lambda: (_ for _ in ()).throw(RuntimeError("no current step"))
    selected = []

    def select_step(name, preferred_step):
        selected.append(preferred_step)
        return preferred_step, "OnNodes"

    viewer._select_step_and_location_for_result = select_step
    viewer._load_cached_values = lambda name, step, location: ("Scalar", {1: np.array([4.0])})

    report = viewer._build_node_report(0.0, 0.0, 1, 0.0, 0.0, 0.0)

    assert selected == [5]
    assert "PRESSURE: 4 [step 5, OnNodes]" in report


def test_load_solution_field_handles_locations_unknown_nodes_and_reductions(tmp_path, monkeypatch):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    solution = viz.load_solution_data("first", "first", base_dir=tmp_path)

    fallback_field = viz.load_solution_field(
        solution, "PRESSURE", "latest", "OnGaussPoints", "magnitude", "trace"
    )
    assert fallback_field["location"] == "OnNodes"
    assert fallback_field["scalar"][0] == 10.0

    gauss_field = viz.load_solution_field(
        solution, "STRESS", "latest", None, "magnitude", "trace"
    )
    assert gauss_field["location"] == "OnGaussPoints"
    assert np.all(np.isfinite(gauss_field["scalar"]))

    vector_field = viz.load_solution_field(
        solution, "DISPLACEMENT", "latest", None, "y", "trace"
    )
    assert vector_field["result_type"] == "Vector"
    assert vector_field["scalar"][0] == 4.0

    original_load_values = viz.load_values
    monkeypatch.setattr(
        viz,
        "load_values",
        lambda *args: ("Scalar", {999: np.array([42.0])}),
    )
    try:
        unknown_field = viz.load_solution_field(
            solution, "PRESSURE", "latest", "OnNodes", "magnitude", "trace"
        )
    finally:
        monkeypatch.setattr(viz, "load_values", original_load_values)
    assert np.all(np.isnan(unknown_field["scalar"]))
