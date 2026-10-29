import builtins
import sys

import pytest

from conftest import make_single_viewer, write_mesh, write_results
from kratos_visualization_tools.gid import cli


@pytest.mark.parametrize("arguments", [
    ["--compare", "other"], ["--list"], ["--extract-json", "missing.json"],
    ["--save", "plot.png"], ["--vmin", "0"], ["--vmax", "1"],
])
def test_gui_rejections(arguments, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gid", "--gui", *arguments])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "--gui" in capsys.readouterr().err


def test_missing_qt_actionable(tmp_path, monkeypatch, capsys):
    mesh, result = tmp_path / "model.post.msh", tmp_path / "model.post.res"
    write_mesh(mesh)
    write_results(result)
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "qt_viewer":
            raise ModuleNotFoundError("No module named PySide6", name="PySide6")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    monkeypatch.setattr(sys, "argv", ["gid", str(mesh), "--gui"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "pip install '.[gui]'" in capsys.readouterr().err


def test_gui_dispatch(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from kratos_visualization_tools.gid import qt_viewer
    mesh, result = tmp_path / "model.post.msh", tmp_path / "model.post.res"
    write_mesh(mesh)
    write_results(result)
    calls = []
    monkeypatch.setattr(qt_viewer, "render_plot_gui", lambda **options: calls.append(options))
    monkeypatch.setattr(sys, "argv", ["gid", "--folder", str(tmp_path), "--gui",
                                    "--result", "PRESSURE", "--colormap", "plasma",
                                    "--save", "ignored.png", "--interactive"])
    cli.main()
    assert calls[0]["initial_colormap"] == "plasma"
    assert len(calls[0]["solution_paths"]) == 1


@pytest.mark.parametrize("arguments", [["--list"], ["--save", "plot.png"], []])
def test_non_gui_routes_never_import_qt(tmp_path, monkeypatch, arguments):
    mesh, result = tmp_path / "model.post.msh", tmp_path / "model.post.res"
    write_mesh(mesh)
    write_results(result)
    original = builtins.__import__

    def forbid(name, *args, **kwargs):
        assert name != "qt_viewer" and not name.startswith("PySide6")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid)
    monkeypatch.setattr(cli, "render_plot_interactive", lambda **kwargs: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["gid", str(mesh), "--result", "PRESSURE", *arguments])
    cli.main()
    if "--save" in arguments:
        assert (tmp_path / "plot.png").is_file()


def test_strict_shared_exports(tmp_path):
    viewer = make_single_viewer(tmp_path, show_profile=True, profile_nodes={1, 3})
    try:
        viewer.profile_y_data = viewer.profile_values = None
        with pytest.raises(ValueError, match="No profile"):
            viewer.export_profile_data(tmp_path / "profile.csv", strict=True)
        with pytest.raises(ValueError, match="requires gradient"):
            viewer.export_current_field_data(tmp_path / "gradient.csv", strict=True)
        viewer.show_profile = True
        viewer.update_plot()
        path = tmp_path / "profile.csv"
        assert viewer.export_profile_data(path, strict=True) == path
        with pytest.raises(OSError):
            viewer.export_profile_data(tmp_path / "missing" / "profile.csv", strict=True)
        viewer.show_gradient = True
        viewer.update_plot()
        path = tmp_path / "gradient.csv"
        assert viewer.export_current_field_data(path, strict=True) == path
        with pytest.raises(OSError):
            viewer.export_current_field_data(tmp_path / "missing" / "gradient.csv", strict=True)
    finally:
        import matplotlib.pyplot as plt
        plt.close(viewer.fig)
