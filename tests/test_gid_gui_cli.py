import builtins
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

from conftest import make_single_viewer, write_mesh, write_results
from kratos_visualization_tools.gid import cli


@pytest.mark.parametrize("arguments", [
    ["--list"], ["--extract-json", "missing.json"],
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


@pytest.mark.parametrize("flag", ["--compare", "--comparison"])
@pytest.mark.parametrize("from_settings", [False, True])
def test_comparison_gui_dispatch(tmp_path, monkeypatch, flag, from_settings):
    import json
    pytest.importorskip("PySide6")
    from kratos_visualization_tools.gid import qt_comparison

    for name in ("first", "second"):
        write_mesh(tmp_path / f"{name}.post.msh")
        write_results(tmp_path / f"{name}.post.res")
    calls = []
    monkeypatch.setattr(qt_comparison, "render_comparison_gui",
                        lambda **options: calls.append(options))
    monkeypatch.setattr(cli, "render_comparison",
                        lambda **options: pytest.fail("Legacy renderer called"))
    if from_settings:
        settings = tmp_path / "comparison.json"
        settings.write_text(json.dumps({
            "mode": "comparison",
            "solutions": [{"mesh": str(tmp_path / f"{name}.post.msh")}
                          for name in ("first", "second")],
            "result": "PRESSURE", "vmin": -1, "vmax": 8,
            "show_mesh": True, "show_isolines": False,
        }))
        arguments = ["--settings", str(settings)]
    else:
        arguments = [str(tmp_path / "first"), flag, str(tmp_path / "second"),
                     "--result", "PRESSURE", "--vmin", "-1", "--vmax", "8"]
    monkeypatch.setattr(sys, "argv", ["gid", "--gui", *arguments])
    cli.main()
    assert len(calls) == 1
    assert calls[0]["vmin"] == -1 and calls[0]["vmax"] == 8
    assert calls[0]["initial_result"] == "PRESSURE"
    assert calls[0]["out_path"] is None
    assert calls[0]["initial_show_mesh"] == from_settings
    assert calls[0]["initial_show_isolines"] != from_settings


def test_comparison_missing_qt_actionable(tmp_path, monkeypatch, capsys):
    for name in ("first", "second"):
        write_mesh(tmp_path / f"{name}.post.msh")
        write_results(tmp_path / f"{name}.post.res")
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "qt_comparison":
            raise ModuleNotFoundError("No module named PySide6", name="PySide6")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    monkeypatch.setattr(sys, "argv", [
        "gid", str(tmp_path / "first"), "--compare", str(tmp_path / "second"), "--gui",
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "pip install '.[gui]'" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--list"], ["--save", "plot.png"], []])
def test_non_gui_comparison_never_imports_qt(tmp_path, monkeypatch, arguments):
    for name in ("first", "second"):
        write_mesh(tmp_path / f"{name}.post.msh")
        write_results(tmp_path / f"{name}.post.res")
    original = builtins.__import__

    def forbid(name, *args, **kwargs):
        assert name not in ("qt_viewer", "qt_comparison") and not name.startswith("PySide6")
        return original(name, *args, **kwargs)

    calls = []
    monkeypatch.setattr(builtins, "__import__", forbid)
    monkeypatch.setattr(cli, "render_comparison", lambda **options: calls.append(options))
    monkeypatch.setattr(sys, "argv", [
        "gid", str(tmp_path / "first"), "--compare", str(tmp_path / "second"), *arguments,
    ])
    cli.main()
    assert len(calls) == (0 if "--list" in arguments else 1)
    if calls:
        assert (calls[0]["out_path"] is not None) == ("--save" in arguments)


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
