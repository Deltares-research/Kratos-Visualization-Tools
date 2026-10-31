"""Offscreen desktop comparison tests using the real shared rendering engine."""

import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from conftest import write_mesh, write_results, write_second_mesh, write_second_results
from kratos_visualization_tools.gid.comparison import ComparisonViewer
from kratos_visualization_tools.gid.qt_comparison import (
    GidComparisonWindow, render_comparison_gui,
)
from kratos_visualization_tools.gid.results import load_solution_data


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def options(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    return dict(
        first=load_solution_data("first", "first", base_dir=tmp_path),
        second=load_solution_data("second", "second", base_dir=tmp_path),
        initial_result="PRESSURE", initial_step="latest", initial_location=None,
        component="magnitude", tensor_component_mode="trace", profile_x=0.0,
        style="background+isolines", vmin=None, vmax=None,
    )


@pytest.fixture
def window(app, options):
    before = plt.get_fignums()
    window = GidComparisonWindow(**options)
    window.show()
    app.processEvents()
    assert plt.get_fignums() == before
    yield window
    window.close()
    app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def test_comparison_controls_and_layout(window, app):
    v = window.viewer
    assert window.result.count() == 2
    assert [window.step.itemText(i) for i in range(window.step.count())] == ["latest", "1"]
    window.step.setCurrentText("1")
    assert [field["step"] for field in v.fields] == [1, 1]
    window.slider.setValue(0)
    assert [field["step"] for field in v.fields] == [2, 1]
    window.result.setCurrentText("DISPLACEMENT")
    assert window.component.isEnabled()
    window.component.setCurrentText("x")
    assert v.current_component == "x"
    for style in v.styles:
        window.style.setCurrentText(style)
        assert v.current_style == style
    window.colormap.setCurrentText("plasma")
    window.levels.setValue(8)
    assert v.current_colormap == "plasma" and v.contour_levels == 8
    for name in ("show_mesh", "show_domain_edge", "show_isolines"):
        previous = getattr(v, name)
        window.checks[name].click()
        assert getattr(v, name) != previous
    window.checks["show_profile"].click()
    assert v.ax1 is not None and v.ax2 is not None
    window.checks["show_profile"].click()
    assert v.ax1 is None and v.ax2 is None
    window.profile_x.setFocus()
    QTest.keyClicks(window.profile_x, "npq")
    assert v.current_result == "DISPLACEMENT" and window.isVisible()
    app.processEvents()


def test_comparison_scale_profile_inspection_and_saves(window, tmp_path, monkeypatch):
    v = window.viewer
    window.minimum.setText("-2")
    window.maximum.setText("8")
    window.apply_scale()
    assert (v.vmin, v.vmax) == (-2, 8)
    window.automatic_scale()
    assert v.requested_vmin is None and v.requested_vmax is None
    window.profile_path.setText("0,0;0,1")
    window.apply_profile()
    assert v.profile_path == ((0.0, 0.0), (0.0, 1.0))
    window.checks["show_profile"].click()
    assert v.ax2 is not None
    window.inspection.setCurrentText("idw")
    window.coordinate.setText("0,0")
    window.inspect_typed()
    assert "PRESSURE" in window.report.toPlainText()
    assert "idw" in window.report.toPlainText()
    assert v.info_fig is None
    window.report.clear()
    event = SimpleNamespace(button=1, inaxes=v.ax0, xdata=0.0, ydata=0.0)
    window.toolbar.pan()
    window.inspect_click(event)
    assert window.report.toPlainText() == ""
    window.toolbar.pan()
    window.inspect_click(event)
    assert "PRESSURE" in window.report.toPlainText()
    for kind, suffix in (("image", ".png"), ("settings", ".json")):
        path = tmp_path / f"comparison{suffix}"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            lambda *args, p=path: (str(p), ""))
        window.save_file(kind)
        assert path.stat().st_size > 0
    settings = json.loads((tmp_path / "comparison.json").read_text())
    assert settings["mode"] == "comparison"
    assert settings["inspection_method"] == "idw"
    assert settings["profile_path"] == [[0.0, 0.0], [0.0, 1.0]]


def test_comparison_errors_are_visible_and_restore_selection(window, monkeypatch):
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *args: warnings.append(args[2]))
    window.minimum.setText("nan")
    window.maximum.setText("2")
    window.run_action(window.apply_scale)
    assert "finite" in warnings[-1]
    assert window.viewer.requested_vmin is None
    window.profile_path.setText("bad")
    window.run_action(window.apply_profile)
    assert "Profile path" in warnings[-1]
    window.coordinate.setText("nan,0")
    window.run_action(window.inspect_typed)
    assert "finite" in warnings[-1]
    window.run_action(lambda: window.select_step("999"))
    assert "Step 999" in warnings[-1]
    assert window.viewer.current_step_str == "latest"
    assert [field["step"] for field in window.viewer.fields] == [2, 1]


def test_comparison_matches_legacy_and_lifecycle(app, options):
    legacy = ComparisonViewer(**options)
    window = render_comparison_gui(**options)
    try:
        np.testing.assert_allclose(
            legacy.difference_field["scalar"], window.viewer.difference_field["scalar"],
            equal_nan=True)
        assert (legacy.vmin, legacy.vmax) == (window.viewer.vmin, window.viewer.vmax)
        assert window in app._gid_windows
    finally:
        plt.close(legacy.fig)
        window.close()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    assert window not in app._gid_windows


def test_tensor_selection(app, options):
    with options["second"].res_path.open("a") as stream:
        stream.write(
            'Result "STRESS" "Load" 1 Matrix OnGaussPoints "GP"\n'
            "Values\n20 1.0 2.0 3.0 4.0 5.0 6.0\nEnd Values\n")
    options["second"] = load_solution_data(
        str(options["second"].msh_path), "second")
    window = GidComparisonWindow(**options)
    try:
        window.result.setCurrentText("STRESS")
        assert window.tensor.isEnabled() and not window.component.isEnabled()
        window.tensor.setCurrentText("xx")
        assert window.viewer.current_tensor_component == "xx"
        assert window.location.findText("OnGaussPoints") >= 0
    finally:
        window.close()
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("minimum,maximum", [(float("nan"), 1), (2, 1)])
def test_invalid_initial_scale_is_reported(app, options, minimum, maximum):
    options.update(vmin=minimum, vmax=maximum)
    with pytest.raises(ValueError, match="Color range"):
        GidComparisonWindow(**options)
