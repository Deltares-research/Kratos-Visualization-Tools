"""Offscreen tests for the GiD workflow chooser."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtWidgets

from conftest import write_mesh, write_results, write_second_mesh, write_second_results
from kratos_visualization_tools.gid import cli
from kratos_visualization_tools.gid.qt_start import GidStartWindow


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def start_window(app):
    window = GidStartWindow()
    window.show()
    app.processEvents()
    yield window
    window.close()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_chooser_exposes_three_workflows(start_window):
    assert start_window.single_button.text() == "View one solution"
    assert start_window.compare_button.text() == "Compare two solutions"
    assert start_window.folder_button.text() == "Browse a results folder"


def test_cli_opens_chooser_without_solution(monkeypatch, app):
    from kratos_visualization_tools.gid import qt_start

    calls = []
    monkeypatch.setattr(qt_start, "render_gid_start_gui",
                        lambda defaults: calls.append(defaults))
    monkeypatch.setattr(sys, "argv", ["gid", "--gui", "--profile"])
    cli.main()
    assert calls == [{
        "result": "WATER_PRESSURE", "step": "latest", "location": None,
        "component": "magnitude", "tensor_component": "trace",
        "profile_x": 18.0, "profile_path": None, "profile": True,
        "isolines": False, "isoline_labels": False, "contour_levels": 12,
        "colormap": None, "comparison_style": "background+isolines",
        "show_mesh": False, "show_gradient": False,
        "show_deformation": False, "deformation_scale": 1.0,
    }]


def test_single_file_picker_launches_viewer(tmp_path, start_window, monkeypatch):
    from kratos_visualization_tools.gid import qt_start

    mesh, result = tmp_path / "model.post.msh", tmp_path / "model.post.res"
    write_mesh(mesh)
    write_results(result)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *args: (str(mesh), ""))
    calls = []
    monkeypatch.setattr(qt_start, "render_plot_gui", lambda **options: calls.append(options))
    start_window.single_button.click()
    assert calls[0]["res_path"] == result
    assert calls[0]["initial_result"] == "DISPLACEMENT"


def test_comparison_pickers_launch_comparison_window(tmp_path, start_window, monkeypatch):
    from kratos_visualization_tools.gid import qt_start

    first_mesh, first_result = tmp_path / "first.post.msh", tmp_path / "first.post.res"
    second_mesh, second_result = tmp_path / "second.post.msh", tmp_path / "second.post.res"
    write_mesh(first_mesh)
    write_results(first_result)
    write_second_mesh(second_mesh)
    write_second_results(second_result)
    selected = iter((str(first_mesh), str(second_mesh)))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *args: (next(selected), ""))
    calls = []
    monkeypatch.setattr(qt_start, "render_comparison_gui",
                        lambda **options: calls.append(options))
    start_window.compare_button.click()
    assert len(calls) == 1
    assert calls[0]["first"].msh_path == first_mesh
    assert calls[0]["second"].msh_path == second_mesh
    assert calls[0]["initial_result"] == "DISPLACEMENT"


def test_folder_picker_discovers_pairs(tmp_path, start_window, monkeypatch):
    from kratos_visualization_tools.gid import qt_start

    mesh, result = tmp_path / "model.post.msh", tmp_path / "model.post.res"
    write_mesh(mesh)
    write_results(result)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        lambda *args: str(tmp_path))
    calls = []
    monkeypatch.setattr(qt_start, "render_plot_gui", lambda **options: calls.append(options))
    start_window.folder_button.click()
    assert calls[0]["solution_paths"] == [(mesh.resolve(), result.resolve())]


def test_empty_folder_reports_actionable_error(tmp_path, start_window, monkeypatch):
    errors = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        lambda *args: str(tmp_path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *args: errors.append(args[2]))
    start_window.folder_button.click()
    assert "No matching" in errors[0]
