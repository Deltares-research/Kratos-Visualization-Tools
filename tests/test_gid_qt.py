"""Real Qt integration tests; optional dependencies are not needed by other tests."""

import json
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.pyplot as plt
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from conftest import write_mesh, write_results, write_second_mesh, write_second_results
from kratos_visualization_tools.gid.parsing import parse_post_msh, scan_result_headers
from kratos_visualization_tools.gid.qt_viewer import GidMainWindow, render_plot_gui


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def options(tmp_path):
    mesh, result = tmp_path / "model.post.msh", tmp_path / "model.post.res"
    write_mesh(mesh)
    write_results(result)
    nodes, triangles, elements, lines, _ = parse_post_msh(mesh)
    return dict(
        headers=scan_result_headers(result), res_path=result, all_nodes=nodes,
        triangles=triangles, surface_elements=elements, line_meshes=lines,
        profile_nodes={1, 3}, initial_result="PRESSURE", initial_step="latest",
        initial_location=None, profile_x=0.0, x_tol=0.35,
        initial_tensor_component="trace", active_filter=None,
    )


@pytest.fixture
def window(app, options):
    before = plt.get_fignums()
    window = GidMainWindow(**options)
    window.show()
    app.processEvents()
    assert plt.get_fignums() == before
    yield window
    window.close()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_selectors_and_controls(window, app):
    v = window.viewer
    assert window.step.count() == 2
    assert window.step.currentText() == "2"
    assert not window.component.isEnabled()
    assert not window.tensor.isEnabled()
    assert not window.scale.isEnabled()
    window.slider.setValue(0)
    assert v.get_current_step() == 1
    window.checks["show_mesh"].click()
    assert v.show_mesh
    window.checks["show_profile"].click()
    assert v.show_profile and v.ax1 is not None
    assert window.profile_export.isEnabled()
    window.checks["show_profile"].click()
    assert v.ax1 is None and not window.profile_export.isEnabled()
    window.result.setCurrentText("DISPLACEMENT")
    assert window.component.isEnabled()
    assert window.scale.isEnabled()
    window.component.setCurrentText("y")
    window.scale.setValue(2.5)
    window.checks["show_deformation"].click()
    assert v.current_component == "y" and v.deformation_scale == 2.5
    assert v.show_deformation
    window.result.setCurrentText("STRESS")
    assert window.tensor.isEnabled() and not window.component.isEnabled()
    assert not v.show_deformation
    assert window.location.currentText() == "OnGaussPoints"
    window.tensor.setCurrentText("xx")
    assert v.current_tensor_component == "xx"
    window.colormap.setCurrentText("plasma")
    window.levels.setValue(8)
    assert v.current_colormap == "plasma" and v.contour_levels == 8
    assert not window.checks["show_isoline_labels"].isEnabled()
    window.checks["show_isolines"].click()
    assert window.checks["show_isoline_labels"].isEnabled()
    app.processEvents()


def test_control_panel_is_compact_with_long_result_names(window, app):
    window.result.addItem("A_VERY_LONG_RESULT_NAME_" * 8)
    app.processEvents()
    panel = window.findChild(QtWidgets.QScrollArea)
    assert panel.width() == 240
    assert panel.horizontalScrollBar().maximum() == 0
    assert window.viewer.canvas.width() > panel.width()


def test_profile_gradient_and_files(window, tmp_path, monkeypatch):
    window.checks["show_profile"].click()
    window.profile_x.setText("0.25")
    window.profile_path.setText("0,0;0,1")
    window.apply_profile()
    window.sync_controls()
    assert window.viewer.profile_path == ((0.0, 0.0), (0.0, 1.0))
    for kind, suffix in (("image", ".png"), ("settings", ".json"), ("profile", ".csv")):
        path = tmp_path / (kind + suffix)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args, p=path: (str(p), ""))
        window.save_file(kind)
        assert path.stat().st_size > 0
        assert "Saved" in window.statusBar().currentMessage()
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["show_profile"]
    assert settings["profile_path"] == [[0.0, 0.0], [0.0, 1.0]]
    window.checks["show_gradient"].click()
    assert window.gradient_export.isEnabled()
    path = tmp_path / "gradient.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window.save_file("gradient")
    assert "gradient_x" in path.read_text()


def test_errors_are_visible_and_no_false_success(window, tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.profile_path.setText("bad")
    window.run_action(window.apply_profile)
    assert "Profile path" in warnings[-1]
    window.coordinate.setText("nan,2")
    window.run_action(window.inspect_typed)
    assert "finite" in warnings[-1]
    path = tmp_path / "missing" / "profile.csv"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window.run_action(lambda: window.save_file("profile"))
    assert "No profile" in warnings[-1]
    window.checks["show_profile"].click()
    window.run_action(lambda: window.save_file("profile"))
    assert "Saved" not in window.statusBar().currentMessage()
    assert not path.exists()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    old_message = window.statusBar().currentMessage()
    window.save_file("settings")
    assert window.statusBar().currentMessage() == old_message


def test_inspection_and_toolbar(window, monkeypatch):
    event = SimpleNamespace(button=1, inaxes=window.viewer.ax0, xdata=0.0, ydata=0.0)
    window.toolbar.pan()
    window.inspect_click(event)
    assert window.report.toPlainText() == ""
    window.toolbar.pan()
    window.inspect_click(event)
    assert "PRESSURE" in window.report.toPlainText()
    window.profile_path.setFocus()
    old_result = window.viewer.current_result
    QTest.keyClicks(window.profile_path, "npq")
    assert window.viewer.current_result == old_result
    assert window.isVisible()
    assert window.viewer.info_fig is None
    window.coordinate.setText("0,0")
    window.inspect_typed()
    assert "PRESSURE" in window.report.toPlainText()


def test_folder_play_and_lifecycle(app, options, tmp_path):
    second_mesh, second_result = tmp_path / "second.post.msh", tmp_path / "second.post.res"
    write_second_mesh(second_mesh)
    write_second_results(second_result)
    options["solution_paths"] = (
        (tmp_path / "model.post.msh", options["res_path"]), (second_mesh, second_result))
    window = GidMainWindow(**options)
    window.solution.setCurrentIndex(1)
    assert window.viewer.res_path == second_result
    assert window.step.count() == 1
    window.solution.setCurrentIndex(0)
    window.toggle_play()
    assert window.timer.isActive()
    QTest.qWait(300)
    window.sync_controls()
    assert window.viewer.res_path == second_result
    window.toggle_play()
    assert not window.timer.isActive()
    window.toggle_play()
    window.close()
    assert not window.timer.isActive()
    assert window.viewer.animation is None
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_location_switching(app, options, monkeypatch):
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    with options["res_path"].open("a") as stream:
        stream.write(
            'Result "PRESSURE" "Load" 3 Scalar OnGaussPoints "GP"\n'
            "Values\n10 3.0\nEnd Values\n")
    options["headers"] = scan_result_headers(options["res_path"])
    window = GidMainWindow(**options)
    window.location.setCurrentText("OnNodes")
    assert window.step.count() == 2
    assert window.viewer.get_available_steps() == [1, 2]
    window.run_action(lambda: window.move_step(1))
    assert window.viewer.get_current_step() == 1
    window.run_action(lambda: window.move_step(-1))
    assert window.viewer.get_current_step() == 2
    window.run_action(window.toggle_play)
    assert window.timer.isActive()
    QTest.qWait(300)
    assert window.viewer.get_current_step() == 1
    assert window.viewer.current_location == "OnNodes"
    window.location.setCurrentText("OnGaussPoints")
    assert not window.timer.isActive()
    assert window.step.count() == 1
    assert window.step.currentText() == "3"
    assert window.viewer.current_location == "OnGaussPoints"
    assert window.viewer.get_available_steps() == [3]
    window.run_action(lambda: window.move_step(1))
    window.run_action(lambda: window.move_step(-1))
    window.run_action(window.toggle_play)
    QTest.qWait(300)
    assert window.viewer.get_current_step() == 3
    assert not window.timer.isActive()
    assert not warnings
    window.close()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("attribute, invalid", [
    ("current_result", "UNAVAILABLE"), ("current_step_str", "999"),
])
def test_failed_selection_restores_operable_controls(window, monkeypatch, attribute, invalid):
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    previous = (window.viewer.current_result, window.viewer.get_current_step())
    window.run_action(lambda: window.change(attribute, invalid))
    assert warnings and invalid in warnings[-1]
    assert window.statusBar().currentMessage() == warnings[-1]
    assert (window.viewer.current_result, window.viewer.get_current_step()) == previous
    assert window.result.currentText() == previous[0]
    assert window.step.currentText() == str(previous[1])
    window.run_action(lambda: window.move_step(1))
    assert window.viewer.get_current_step() == 1
    window.checks["show_mesh"].click()
    assert window.viewer.show_mesh


def test_renderer_existing_application(app, options):
    window = render_plot_gui(**options)
    assert window.isVisible()
    assert window in app._gid_windows
    window.close()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert window not in app._gid_windows


def test_initial_options(app, options):
    options.update(initial_component="x", initial_show_mesh=True, show_profile=True,
                   initial_colormap="plasma", initial_show_gradient=True,
                   show_isolines=True, contour_levels=6, show_isoline_labels=True)
    window = GidMainWindow(**options)
    assert window.component.currentText() == "x"
    assert window.checks["show_mesh"].isChecked()
    assert window.checks["show_gradient"].isChecked()
    assert window.checks["show_profile"].isChecked()
    assert window.colormap.currentText() == "plasma"
    assert window.levels.value() == 6
    window.close()


def test_real_cli_launch_with_saved_settings(app, options, tmp_path, monkeypatch):
    from kratos_visualization_tools.gid import cli
    settings_path = tmp_path / "initial.json"
    settings_path.write_text(json.dumps({
        "mode": "single", "result": "DISPLACEMENT", "step": "2",
        "component": "y", "show_mesh": True, "show_profile": True,
        "colormap": "plasma", "show_deformation": True, "deformation_scale": 4.0,
    }))
    monkeypatch.setattr(sys, "argv", [
        "gid", str(tmp_path / "model.post.msh"), "--gui", "--settings", str(settings_path)])
    cli.main()
    window = app._gid_windows[-1]
    assert window.isVisible()
    assert window.component.currentText() == "y"
    assert window.checks["show_mesh"].isChecked()
    assert window.checks["show_profile"].isChecked()
    assert window.checks["show_deformation"].isChecked()
    assert window.scale.value() == 4.0
    window.close()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
