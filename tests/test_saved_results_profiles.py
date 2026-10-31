import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kratos_visualization_tools.saved_results import cli, session
from kratos_visualization_tools.saved_results.config import run
from kratos_visualization_tools.saved_results.viewer import FigureViewer


def profile_config(tmp_path, name="first", xlabel="Pressure"):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "profile.csv").write_text("coordinate,value\n0,1\n1,2\n")
    config = {
        "data_dir": ".", "output_dir": "plots",
        "metadata": {"mode": "single"},
        "figures": [{
            "type": "single", "name": "profile.svg",
            "xlabel": xlabel, "ylabel": "y",
            "series": [{
                "source": "csv", "file": "profile.csv", "label": name,
                "x_column": "value", "y_column": "coordinate",
            }],
        }],
    }
    path = directory / "plot_config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def viewer(tmp_path):
    window = FigureViewer(run(profile_config(tmp_path), show=True))
    yield window
    plt.close(window.figure)


def test_csv_import_overlays_exact_data_and_updates_saved_figure(viewer, tmp_path):
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    first.write_text("y_coordinate,value,gradient\n1,8,0\n0,4,0\n")
    second.write_text("series,coordinate,value\nsolution_1,0,3\nsolution_2,1,6\n")
    pending = viewer.pending_figures[0]
    before = pending.preview.copy()
    viewer.import_profiles([first, second])
    lines = pending.figure.axes[0].lines
    assert len(lines) == 4
    np.testing.assert_array_equal(lines[1].get_xdata(), [4, 8])
    np.testing.assert_array_equal(lines[1].get_ydata(), [0, 1])
    assert [line.get_label() for line in lines] == [
        "first", "one", "two: solution_1", "two: solution_2"]
    assert not np.array_equal(before, pending.preview)
    assert len(pending.figure.legends[0].get_texts()) == 4
    viewer.save()
    assert pending.path.is_file()
    assert "two: solution_2" in pending.path.read_text()


def test_bad_csv_does_not_partially_import(viewer, tmp_path):
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    good.write_text("y_coordinate,value\n0,3\n")
    bad.write_text("y_coordinate,value\n0,nan\n")
    with pytest.raises(ValueError, match="Non-finite"):
        viewer.import_profiles([good, bad])
    assert len(viewer.pending_figures[0].figure.axes[0].lines) == 1


def test_missing_column_is_reported(viewer, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n0,3\n")
    with pytest.raises(KeyError):
        viewer.import_profiles([bad])


def test_import_button_dialog_and_cancellation(viewer, tmp_path, monkeypatch):
    pytest.importorskip("tkinter")
    from tkinter import filedialog, messagebox

    viewer.figure.canvas.manager.window = object()
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda *args, **kw: errors.append(args))
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **kw: ())
    viewer.add_profile()
    assert len(viewer.pending_figures[0].figure.axes[0].lines) == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("bad\n")
    monkeypatch.setattr(filedialog, "askopenfilenames", lambda **kw: (str(bad),))
    viewer.add_profile()
    assert errors and errors[0][0] == "Add profile"
    del viewer.figure.canvas.manager.window


def test_forwarded_profiles_merge_but_different_quantities_stay_separate(viewer, tmp_path):
    identity = viewer.figure
    viewer.add_configuration(profile_config(tmp_path, "second"))
    assert viewer.figure is identity
    assert len(viewer.pending_figures) == 1
    assert len(viewer.pending_figures[0].figure.axes[0].lines) == 2
    viewer.add_configuration(profile_config(tmp_path, "displacement", "Displacement"))
    assert len(viewer.pending_figures) == 2
    viewer._select(SimpleNamespace(inaxes=viewer._content_axes[1]))
    assert viewer.index == 1
    csv = tmp_path / "third.csv"
    csv.write_text("coordinate,value\n0,12\n")
    viewer.import_profiles([csv])
    assert len(viewer.pending_figures[1].figure.axes[0].lines) == 2


def test_comparison_bundle_adds_both_curves_and_separate_difference(viewer, tmp_path):
    config = profile_config(tmp_path, "comparison")
    data = json.loads(config.read_text())
    data["metadata"]["mode"] = "comparison"
    figure = data["figures"][0]
    figure["series"].append({**figure["series"][0], "label": "comparison_2"})
    data["figures"].append({
        **figure, "name": "difference.svg", "xlabel": "Pressure difference",
        "series": [{**figure["series"][0], "label": "difference"}],
    })
    config.write_text(json.dumps(data))
    viewer.add_configuration(config)
    assert len(viewer.pending_figures) == 2
    assert [line.get_label() for line in viewer.pending_figures[0].figure.axes[0].lines] == [
        "first", "comparison", "comparison_2"]
    assert viewer.pending_figures[1].path.name == "difference.svg"
    assert viewer.pending_figures[1].figure.axes[0].get_xlabel() == "Pressure difference"
    viewer.save()
    assert all(pending.path.is_file() for pending in viewer.pending_figures)


def test_duplicate_import_labels_remain_distinguishable(viewer, tmp_path):
    csv = tmp_path / "first.csv"
    csv.write_text("coordinate,value\n0,12\n")
    viewer.import_profiles([csv, csv])
    assert [line.get_label() for line in viewer.pending_figures[0].figure.axes[0].lines] == [
        "first", "first (2)", "first (3)"]


def test_multiaxis_import_requires_single_axis_selection(viewer, tmp_path):
    viewer.pending_figures[0].figure.add_subplot(212)
    csv = tmp_path / "profile.csv"
    csv.write_text("coordinate,value\n0,12\n")
    with pytest.raises(ValueError, match="single-axis"):
        viewer.import_profiles([csv])


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    address = (rf"\\.\pipe\kratos-test-{uuid4().hex}" if os.name == "nt"
               else str(tmp_path / "test.sock"))
    family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
    monkeypatch.setattr(session, "_endpoint", lambda: (address, family, b"test-key"))
    return address


def test_real_local_handoff_reuses_window_and_can_restart(viewer, tmp_path, endpoint):
    config = profile_config(tmp_path, "second")
    owner = session.open_session(config)
    owner.attach(viewer)
    result = []
    client = threading.Thread(target=lambda: result.append(session.open_session(config)))
    client.start()
    try:
        deadline = time.monotonic() + 10
        while client.is_alive() and time.monotonic() < deadline:
            owner.process_requests()
            time.sleep(0.01)
        client.join(timeout=1)
        assert not client.is_alive()
        assert result == [None]
        assert len(viewer.pending_figures) == 1
        assert len(viewer.pending_figures[0].figure.axes[0].lines) == 2
    finally:
        owner.close()
    assert not owner.thread.is_alive()
    restarted = session.open_session(config)
    assert restarted is not None
    restarted.close()


def test_cli_forwarding_does_not_create_another_viewer(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "open_session", lambda *args: None)
    monkeypatch.setattr(cli.plt, "switch_backend",
                        lambda *args: pytest.fail("Must not change backend"))
    monkeypatch.setattr(cli, "run", lambda *args, **kw: pytest.fail("Must not reload"))
    assert cli.main([str(tmp_path / "plot_config.json"), "--show"]) == 0


def test_handoff_errors_are_returned_to_sender(viewer, tmp_path, endpoint):
    owner = session.open_session(tmp_path / "initial.json")
    owner.attach(viewer)
    errors = []

    def send():
        try:
            session.open_session(tmp_path / "missing.json")
        except ValueError as error:
            errors.append(str(error))

    client = threading.Thread(target=send)
    client.start()
    try:
        deadline = time.monotonic() + 10
        while client.is_alive() and time.monotonic() < deadline:
            owner.process_requests()
            time.sleep(0.01)
        client.join(timeout=1)
        assert not client.is_alive()
        assert len(errors) == 1 and "missing.json" in errors[0]
        assert len(viewer.pending_figures[0].figure.axes[0].lines) == 1
    finally:
        owner.close()


@pytest.mark.skipif(os.name != "nt", reason="Requires the Windows desktop Tk backend")
def test_real_gui_subprocess_reuses_window_and_saves_overlay(tmp_path):
    pytest.importorskip("tkinter")
    first = profile_config(tmp_path, "first")
    second = profile_config(tmp_path, "second")
    address = rf"\\.\pipe\kratos-gui-test-{uuid4().hex}"
    ready, result = tmp_path / "ready", tmp_path / "result.json"
    server_code = """
import json
import sys
from pathlib import Path
from kratos_visualization_tools.saved_results import cli, session
from kratos_visualization_tools.saved_results.viewer import FigureViewer
session._endpoint = lambda: (sys.argv[1], "AF_PIPE", b"gui-test")
class CheckedViewer(FigureViewer):
    def __init__(self, pending):
        super().__init__(pending)
        self.identity = self.figure
        root = self.figure.canvas.manager.window
        root.after(100, lambda: Path(sys.argv[3]).write_text("ready"))
        root.after(20000, lambda: cli.plt.close(self.figure))
    def add_configuration(self, *args):
        super().add_configuration(*args)
        self.save()
        lines = self.pending_figures[0].figure.axes[0].lines
        Path(sys.argv[4]).write_text(json.dumps({
            "same_window": self.figure is self.identity,
            "windows": len(cli.plt.get_fignums()),
            "figures": len(self.pending_figures),
            "curves": [line.get_xdata().tolist() for line in lines]
        }))
        self.figure.canvas.manager.window.after(200, lambda: cli.plt.close(self.figure))
cli.FigureViewer = CheckedViewer
sys.exit(cli.main([sys.argv[2], "--show"]))
"""
    client_code = """
import sys
from kratos_visualization_tools.saved_results import cli, session
session._endpoint = lambda: (sys.argv[1], "AF_PIPE", b"gui-test")
sys.exit(cli.main([sys.argv[2], "--show"]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", server_code, address, str(first), str(ready), str(result)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), process.communicate(timeout=10)
        client = subprocess.run(
            [sys.executable, "-c", client_code, address, str(second)],
            capture_output=True, text=True, timeout=15,
        )
        assert client.returncode == 0, client.stderr
        output, error = process.communicate(timeout=10)
        assert process.returncode == 0, output + error
        assert json.loads(result.read_text()) == {
            "same_window": True, "windows": 1, "figures": 1,
            "curves": [[1, 2], [1, 2]],
        }
        saved = first.parent / "plots" / "profile.svg"
        assert saved.is_file()
        assert "first" in saved.read_text() and "second" in saved.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
