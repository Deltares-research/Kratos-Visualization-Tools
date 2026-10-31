import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import (
    make_single_viewer, write_mesh, write_results, write_second_mesh, write_second_results,
)
from kratos_visualization_tools.gid.comparison import ComparisonViewer
from kratos_visualization_tools.gid.profile_bundle import save_profile_bundle
from kratos_visualization_tools.gid.results import load_solution_data
from kratos_visualization_tools.saved_results.config import run


def test_single_bundle_is_portable_and_preserves_current_profile(tmp_path):
    viewer = make_single_viewer(tmp_path, show_profile=True, profile_nodes={1, 3})
    try:
        expected = np.column_stack((viewer.profile_y_data, viewer.profile_values))
        config = save_profile_bundle(viewer, tmp_path)
        data = json.loads(config.read_text())
        assert data["metadata"]["sources"][0]["step"] == viewer.get_current_step()
        assert data["metadata"]["result"] == "PRESSURE"
        assert data["figures"][0]["series"][0]["file"] == "profile.csv"
        with (config.parent / "profile.csv").open() as source:
            rows = list(csv.DictReader(source))
        np.testing.assert_allclose(
            [[float(row["coordinate"]), float(row["value"])] for row in rows], expected)
        moved = tmp_path / "moved bundle"
        shutil.move(str(config.parent), moved)
        (tmp_path / "model.post.msh").unlink()
        (tmp_path / "model.post.res").unlink()
        pending = run(moved / "plot_config.json", show=True)
        assert len(pending) == 1
        np.testing.assert_allclose(pending[0].figure.axes[0].lines[0].get_xdata(),
                                   expected[:, 1])
        result = subprocess.run([
            sys.executable, "-m", "kratos_visualization_tools.saved_results",
            str(moved / "plot_config.json"),
        ], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr + result.stdout
        assert (moved / "plots" / "profile.svg").is_file()
    finally:
        plt.close("all")


def test_comparison_bundle_contains_both_profiles_and_exact_difference(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    viewer = ComparisonViewer(
        first=load_solution_data("first", "first", tmp_path),
        second=load_solution_data("second", "second", tmp_path),
        initial_result="PRESSURE", initial_step="latest", initial_location=None,
        component="magnitude", tensor_component_mode="trace", profile_x=0,
        style="both-filled", vmin=None, vmax=None, profile_path=((0, 0), (0, 1)),
    )
    try:
        config = save_profile_bundle(viewer, tmp_path, include_difference=True)
        metadata = json.loads(config.read_text())["metadata"]
        assert [source["step"] for source in metadata["sources"]] == [2, 1]
        assert metadata["profile_path"] == [[0, 0], [0, 1]]
        first = viewer._get_profile_data(viewer.solutions[0], viewer.fields[0])
        second = viewer._get_profile_data(viewer.solutions[1], viewer.fields[1])
        expected = viewer._difference_profile(first, second)
        with (config.parent / "profile.csv").open() as source:
            rows = list(csv.DictReader(source))
        difference = [row for row in rows if row["series"] == "difference"]
        np.testing.assert_allclose([float(row["value"]) for row in difference],
                                   expected[1], equal_nan=True)
        pending = run(config, show=True)
        assert len(pending) == 2
        assert len(pending[0].figure.axes[0].lines) == 2
        assert len(pending[1].figure.axes[0].lines) == 1
        without = save_profile_bundle(viewer, tmp_path)
        assert len(json.loads(without.read_text())["figures"]) == 1
    finally:
        plt.close("all")


def test_missing_profile_does_not_create_bundle(tmp_path):
    viewer = make_single_viewer(tmp_path)
    try:
        with pytest.raises(ValueError, match="No profile"):
            save_profile_bundle(viewer, tmp_path)
        assert list(tmp_path.glob("profile_*")) == []
    finally:
        plt.close(viewer.fig)


def test_bundle_write_error_is_reported_and_partial_files_removed(tmp_path, monkeypatch):
    viewer = make_single_viewer(tmp_path, show_profile=True, profile_nodes={1, 3})
    original = Path.write_text

    def fail_config(path, *args, **kwargs):
        if path.name == "plot_config.json":
            raise OSError("Cannot write plot configuration")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_config)
    try:
        with pytest.raises(OSError, match="Cannot write"):
            save_profile_bundle(viewer, tmp_path)
        assert list(tmp_path.glob("profile_*")) == []
    finally:
        plt.close(viewer.fig)
