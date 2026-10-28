import json
import sys

import pytest

import visualize_gid_results as viz

from conftest import (
    write_mesh,
    write_results,
    write_second_mesh,
    write_second_results,
)


def write_active_mdpa(path, submodel_name="active", include_entities=True):
    content = [
        f"Begin SubModelPart {submodel_name}\n",
        "Begin SubModelPartNodes\n",
        "1\n",
        "3\n",
        "End SubModelPartNodes\n",
    ]
    if include_entities:
        content.extend([
            "Begin SubModelPartElements\n",
            "10\n",
            "End SubModelPartElements\n",
        ])
    content.append("End SubModelPart\n")
    path.write_text("".join(content), encoding="utf-8")


def test_print_result_summary_reports_grouped_results(capsys):
    headers = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnNodes", None, 2),
        viz.ResultHeader("STRESS", 3, "Matrix", "OnGaussPoints", "GP", 3),
    ]

    viz.print_result_summary(headers)

    output = capsys.readouterr().out
    assert "Available results:" in output
    assert "PRESSURE" in output
    assert "steps=[1..2], count=2" in output
    assert "STRESS" in output
    assert "gp=GP" in output


def test_print_comparison_summary_reports_common_and_unique_results(tmp_path, capsys):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)

    viz.print_comparison_summary(first, second)

    output = capsys.readouterr().out
    assert "Common variables (first and second):" in output
    assert "DISPLACEMENT" in output
    assert "Only in first" in output
    assert "STRESS" in output
    assert "Only in second" in output
    assert "TEMPERATURE" in output


def test_main_lists_single_solution_results(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["visualize_gid_results.py", "model", "--list"])

    viz.main()

    assert "Available results:" in capsys.readouterr().out


def test_main_dispatches_single_static_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_plot_static", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "model", "--result", "PRESSURE", "--save", "output.png", "--isolines", "--profile"],
    )

    viz.main()

    assert captured["result_name"] == "PRESSURE"
    assert captured["show_isolines"]
    assert captured["show_profile"]


def test_main_dispatches_single_interactive_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_plot_interactive", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "model", "--isolines", "--profile"],
    )

    viz.main()

    assert captured["initial_result"] == "WATER_PRESSURE"
    assert captured["show_isolines"]
    assert captured["show_profile"]


def test_main_dispatches_folder_interactive_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    captured = {}
    monkeypatch.setattr(viz, "render_plot_interactive", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--folder", str(tmp_path), "--result", "PRESSURE"],
    )

    viz.main()

    assert len(captured["solution_paths"]) == 2
    assert captured["solution_paths"][0][0].name == "first.post.msh"


def test_main_loads_single_visualization_settings(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    settings_path = tmp_path / "single_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "mode": "single",
                "result": "PRESSURE",
                "step": "1",
                "location": "OnNodes",
                "component": "x",
                "tensor_component": "trace",
                "profile_x": 0.25,
                "x_tol": 0.1,
                "show_gradient": True,
                "show_mesh": True,
                "show_isolines": True,
                "show_profile": True,
                "show_isoline_labels": True,
                "colormap": "plasma",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_plot_interactive", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "model", "--settings", str(settings_path)],
    )

    viz.main()

    assert captured["initial_result"] == "PRESSURE"
    assert captured["initial_step"] == "1"
    assert captured["initial_component"] == "x"
    assert captured["initial_show_gradient"]
    assert captured["initial_show_mesh"]
    assert captured["initial_colormap"] == "plasma"
    assert captured["show_profile"]


def test_main_loads_comparison_visualization_settings(tmp_path, monkeypatch):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    settings_path = tmp_path / "comparison_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "mode": "comparison",
                "solutions": [
                    {"mesh": str(tmp_path / "first.post.msh")},
                    {"mesh": str(tmp_path / "second.post.msh")},
                ],
                "result": "DISPLACEMENT",
                "step": "latest",
                "location": "OnNodes",
                "component": "magnitude",
                "tensor_component": "trace",
                "profile_x": 0.0,
                "style": "difference",
                "inspection_method": "idw",
                "colormap": "cividis",
                "show_profile": True,
                "show_domain_edge": False,
                "vmin": -1.0,
                "vmax": 2.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_comparison", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--settings", str(settings_path)],
    )

    viz.main()

    assert captured["initial_result"] == "DISPLACEMENT"
    assert captured["style"] == "difference"
    assert captured["inspection_method"] == "idw"
    assert captured["colormap"] == "cividis"
    assert captured["show_profile"]
    assert not captured["show_domain_edge"]
    assert captured["vmin"] == -1.0
    assert captured["vmax"] == 2.0


def test_main_dispatches_comparison_and_replaces_non_common_result(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_comparison", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "first", "--compare", "second",
            "--result", "TEMPERATURE", "--comparison-style", "difference",
            "--inspection-method", "idw", "--colormap", "cividis",
        ],
    )

    viz.main()

    assert captured["initial_result"] == "DISPLACEMENT"
    assert captured["style"] == "difference"
    assert captured["inspection_method"] == "idw"
    assert captured["colormap"] == "cividis"
    assert "not common" in capsys.readouterr().out


def test_main_dispatches_extract_json(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "extract.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_export(json_path, csv_out=None):
        captured["json_path"] = json_path
        captured["csv_out"] = csv_out
        return tmp_path / "output.csv"

    monkeypatch.setattr(viz, "export_line_csv_from_json", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--extract-json", str(config_path), "--csv-out", "out.csv"],
    )

    viz.main()

    assert captured["json_path"] == config_path.resolve()
    assert captured["csv_out"] == (tmp_path / "out.csv").resolve()
    assert "Saved CSV" in capsys.readouterr().out


@pytest.mark.parametrize("show", [False, True])
def test_main_plots_the_extracted_csv_with_plot_saved_results(tmp_path, monkeypatch, show):
    import plot_saved_results

    extract_path = tmp_path / "extract.json"
    extract_path.write_text("{}", encoding="utf-8")
    plot_config_path = tmp_path / "plots.json"
    plot_config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "export_line_csv_from_json", lambda json_path, csv_out=None: tmp_path / "out.csv")
    plot_calls = []
    monkeypatch.setattr(plot_saved_results, "main", lambda argv: plot_calls.append(argv) or 0)
    argv = ["visualize_gid_results.py", "--extract-json", str(extract_path), "--plot-config", str(plot_config_path)]
    if show:
        argv.append("--plot-show")
    monkeypatch.setattr(sys, "argv", argv)

    viz.main()

    expected = [str(plot_config_path.resolve())] + (["--show"] if show else [])
    assert plot_calls == [expected]


def test_main_reports_a_failing_plot_config(tmp_path, monkeypatch):
    import plot_saved_results

    extract_path = tmp_path / "extract.json"
    extract_path.write_text("{}", encoding="utf-8")
    plot_config_path = tmp_path / "plots.json"
    plot_config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "export_line_csv_from_json", lambda json_path, csv_out=None: tmp_path / "out.csv")
    monkeypatch.setattr(plot_saved_results, "main", lambda argv: 1)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--extract-json", str(extract_path), "--plot-config", str(plot_config_path)],
    )

    with pytest.raises(RuntimeError, match="Plotting with"):
        viz.main()


def test_main_reports_a_missing_plot_config(tmp_path, monkeypatch):
    extract_path = tmp_path / "extract.json"
    extract_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "export_line_csv_from_json", lambda json_path, csv_out=None: tmp_path / "out.csv")
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--extract-json", str(extract_path), "--plot-config", str(tmp_path / "missing.json")],
    )

    with pytest.raises(FileNotFoundError, match="Plot config not found"):
        viz.main()


@pytest.mark.parametrize(
    "extra_args",
    [["model", "--plot-config", "plots.json"], ["--extract-json", "extract.json", "--plot-show"]],
)
def test_main_rejects_plot_options_without_their_prerequisites(monkeypatch, extra_args):
    monkeypatch.setattr(sys, "argv", ["visualize_gid_results.py", *extra_args])

    with pytest.raises(SystemExit):
        viz.main()


def test_main_uses_explicit_active_mdpa_filter(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    mdpa_path = tmp_path / "active.mdpa"
    write_active_mdpa(mdpa_path)
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "PRESSURE",
            "--mdpa", str(mdpa_path), "--active", "active", "--save", "out.png",
        ],
    )

    viz.main()

    assert captured["active_filter"].name == "active"
    assert captured["active_filter"].node_ids == {1, 3}


def test_main_uses_automatic_active_mdpa_and_handles_empty_filter(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    write_active_mdpa(tmp_path / "model.mdpa", submodel_name="other")
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "PRESSURE",
            "--active", "missing", "--save", "out.png",
        ],
    )

    viz.main()

    assert captured["active_filter"] is None
    assert "No active filtering will be applied" in capsys.readouterr().out


def test_main_active_filter_requires_an_automatically_located_mdpa(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "PRESSURE",
            "--active", "active", "--save", "out.png",
        ],
    )

    with pytest.raises(FileNotFoundError, match="Could not auto-locate"):
        viz.main()


def test_main_static_projects_gauss_point_results(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "STRESS",
            "--location", "OnGaussPoints", "--save", "stress.png",
        ],
    )

    viz.main()

    assert len(captured["values"]) == 3
    assert "Projecting STRESS" in capsys.readouterr().out


def test_main_static_rejects_empty_result_values(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    (tmp_path / "model.post.res").write_text(
        'Result "EMPTY" "Load" 1 Scalar OnNodes\nEnd Values\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "EMPTY",
            "--save", "empty.png",
        ],
    )

    with pytest.raises(RuntimeError, match="Could not load values"):
        viz.main()


def test_main_passes_profile_path_to_static_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--save", "profile.png",
            "--result", "PRESSURE", "--profile", "--profile-path", "0,0;0.5,0;0.5,0.5",
        ],
    )

    viz.main()

    assert captured["profile_path"] == [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
