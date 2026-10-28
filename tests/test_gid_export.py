import csv
import json

import numpy as np
import pytest

import visualize_gid_results as viz

from conftest import (
    close_viewer_figures,
    make_single_viewer,
    write_mesh,
    write_results,
)


def test_legacy_export_helper_patch_reaches_package_calls(monkeypatch, tmp_path):
    from kratos_visualization_tools import gid

    expected = tmp_path / "patched.csv"
    original = viz.build_output_path
    with monkeypatch.context() as patch:
        patch.setattr(viz, "build_output_path", lambda *args: expected)
        assert gid.build_result_export_path("pressure", "1", "OnNodes", "field") == expected
        assert viz.build_result_export_path("pressure", "1", "OnNodes", "field") == expected
    assert gid.build_output_path is original


def test_interactive_viewer_profile_export(tmp_path, monkeypatch, capsys):
    viewer = make_single_viewer(tmp_path)
    try:
        monkeypatch.chdir(tmp_path)
        viewer.export_profile_data()
        assert "No profile data to export" in capsys.readouterr().out
        viewer._set_profile_data([0.0, 1.0], [2.0, 4.0], profile_x_current=None)
        viewer.export_profile_data()
        output_path = tmp_path / "PRESSURE_latest_OnNodes_magnitude_profile.csv"
        assert output_path.is_file()
        assert output_path.read_text().splitlines() == [
            "y_coordinate,value,gradient", "0.0,2.0,2.0", "1.0,4.0,2.0",
        ]
    finally:
        close_viewer_figures(viewer)


def test_export_current_field_data_and_save_current_plot(tmp_path, monkeypatch, capsys):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)
    viewer = viz.InteractiveViewer(
        headers=headers,
        res_path=result_path,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements,
        line_meshes=line_meshes,
        profile_nodes=set(),
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
    )
    try:
        viewer.export_current_field_data()
        assert "only available while gradient mode is on" in capsys.readouterr().out

        viewer.show_gradient = True
        viewer.export_current_field_data()
        assert "No gradient field is currently available" in capsys.readouterr().out

        viewer.current_field_node_ids = np.array([1, 2])
        viewer.current_field_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        viewer.current_field_source_values = np.array([1.0, np.nan])
        viewer.current_field_display_values = np.array([1.0, 2.0])
        viewer.current_field_grad_x = np.array([2.0, np.nan])
        viewer.current_field_grad_y = np.array([3.0, np.nan])
        viewer.current_field_grad_magnitude = np.array([np.sqrt(13.0), np.nan])
        monkeypatch.chdir(tmp_path)
        viewer.export_current_field_data()
        output_path = tmp_path / "PRESSURE_latest_OnNodes_magnitude_gradient_nodes.csv"
        assert output_path.is_file()
        assert len(output_path.read_text().splitlines()) == 2

        saved_path = viewer.save_current_plot(tmp_path / "current_plot.png")
        assert saved_path.is_file()
        default_saved_path = viewer.save_current_plot()
        assert default_saved_path.resolve() == tmp_path / "PRESSURE_latest_OnNodes_interactive.png"
        assert default_saved_path.is_file()
        settings_path = viewer.save_current_settings(tmp_path / "single_settings.json")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["mode"] == "single"
        assert settings["result"] == "PRESSURE"
        assert settings["show_gradient"]
    finally:
        close_viewer_figures(viewer)


def test_export_line_csv_handles_vector_results(tmp_path):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    config_path = tmp_path / "vector_extract.json"
    config_path.write_text(
        '{"files": ["model"], "result": "DISPLACEMENT", "component": "y", '
        '"line_points": [[0, 0], [0, 1]], "line_samples": 2}',
        encoding="utf-8",
    )

    output_path = viz.export_line_csv_from_json(config_path)

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert [float(row["value"]) for row in rows] == pytest.approx([4.0, 3.333333333333111])
    assert [row["component"] for row in rows] == ["y", "y"]

    gauss_config_path = tmp_path / "gauss_extract.json"
    gauss_config_path.write_text(
        '{"files": ["model"], "result": "STRESS", "location": "OnGaussPoints", '
        '"line": {"start": [0, 0], "end": [0, 1], "samples": 2}}',
        encoding="utf-8",
    )
    gauss_output_path = viz.export_line_csv_from_json(gauss_config_path)
    gauss_rows = list(csv.DictReader(gauss_output_path.open(newline="", encoding="utf-8")))
    assert len(gauss_rows) == 2
    assert [float(row["value"]) for row in gauss_rows] == pytest.approx([1.0, 1.0])


def test_export_line_csv_runs_end_to_end(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    config_path = tmp_path / "extract.json"
    config_path.write_text(
        '{"files": ["model"], "result": "PRESSURE", "step": "latest", '
        '"line": {"start": [0, 0], "end": [0, 1], "samples": 2}}',
        encoding="utf-8",
    )

    output_path = viz.export_line_csv_from_json(config_path)

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [row["location"] for row in rows] == ["OnNodes"] * 2
    assert [float(row["value"]) for row in rows] == [10.0, 12.0]
    assert [float(row["gradient"]) for row in rows] == [2.0, 2.0]
