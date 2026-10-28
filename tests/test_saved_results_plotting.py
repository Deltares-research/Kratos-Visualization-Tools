import json

import pytest

import plot_saved_results as psr


def test_legacy_pending_figures_state_is_live(monkeypatch, tmp_path):
    from kratos_visualization_tools import saved_results

    captured = []

    def make_figures(*args):
        assert psr._pending_figures is saved_results._pending_figures
        psr._pending_figures.append("preview")
        captured.append(psr._pending_figures)

    monkeypatch.setattr(psr, "_make_figures", make_figures)
    assert psr.run(tmp_path / "config.json", show=True) == ["preview"]
    assert captured == [["preview"]]
    assert psr._pending_figures is None
    assert saved_results._pending_figures is None


class TestMinMaxAnnotation:
    def test_reports_extremes_per_series(self):
        annotation = psr._min_max_annotation(
            [psr.DataSeries([(0.0, -1.5), (1.0, 2.5)], "Kratos")]
        )

        assert annotation.splitlines() == [
            f"Max. Kratos = {2.5:12.3e}",
            f"Min. Kratos = {-1.5:12.3e}",
        ]


def write_config(directory, figures, **overrides):
    config = {
        "data_dir": ".",
        "output_dir": "plots",
        "mesh": {"mdpa_file": "model.mdpa"},
        "node_sets": {"wall": {"sub_model_part": "wall"}},
        "stages": ["stage_1", "stage_2"],
        "figures": figures,
    }
    config.update(overrides)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


class TestConfigTemplates:
    def test_expands_defaults_and_series_templates_with_overrides(self):
        config = {
            "figure_defaults": {"type": "profiles_per_stage", "ylabel": "$y$ [m]"},
            "series_templates": {
                "comparison": {"source": "csv", "marker": ".", "color": "blue"}
            },
            "figures": [
                {
                    "name": "plot.svg",
                    "series": [
                        {"template": "comparison", "file": "data.csv", "color": "red"},
                        {"file": "raw.json"},
                    ],
                }
            ],
        }

        assert psr._expand_config(config) == [
            {
                "type": "profiles_per_stage",
                "ylabel": "$y$ [m]",
                "name": "plot.svg",
                "series": [
                    {
                        "source": "csv",
                        "marker": ".",
                        "color": "red",
                        "file": "data.csv",
                    },
                    {"file": "raw.json"},
                ],
            }
        ]

    def test_unknown_series_template_raises_clear_error(self):
        with pytest.raises(KeyError, match="Unknown series template 'missing'"):
            psr._expand_config(
                {
                    "figures": [{"series": [{"template": "missing"}]}],
                }
            )

    def test_expands_figure_template_after_defaults_and_before_overrides(self):
        config = {
            "figure_defaults": {"type": "single", "ylabel": "default"},
            "figure_templates": {
                "history": {
                    "type": "histories_per_node",
                    "node_set": "probes",
                    "ylabel": "template",
                }
            },
            "series_templates": {"kratos": {"label": "Kratos", "marker": "."}},
            "figures": [
                {
                    "template": "history",
                    "name": "history.svg",
                    "ylabel": "figure",
                    "series": [{"template": "kratos", "variable": "PRESSURE"}],
                }
            ],
        }

        assert psr._expand_config(config) == [
            {
                "type": "histories_per_node",
                "ylabel": "figure",
                "node_set": "probes",
                "name": "history.svg",
                "series": [{"label": "Kratos", "marker": ".", "variable": "PRESSURE"}],
            }
        ]

    def test_unknown_figure_template_raises_clear_error(self):
        with pytest.raises(KeyError, match="Unknown figure template 'missing'"):
            psr._expand_config(
                {"figures": [{"template": "missing"}]}
            )


class TestStageDiscovery:
    def test_discovers_unique_stages_and_sorts_them_numerically(self, tmp_path):
        for file_name in (
            "1_Initial_json.json",
            "2_Null_json.json",
            "3_Oude_json.json",
            "3_Zomerkade_json.json",
            "10_End_json.json",
            "1_Initial_nodes.json",
            "3_Oude_nodes.json",
            "10_End_nodes.json",
        ):
            (tmp_path / file_name).write_text("{}")

        figures = psr._expand_config(
            {
                "series_templates": {
                    "json": {"file": "{stage}_json.json"},
                    "nodes": {"file": "{stage}_nodes.json"},
                },
                "figures": [
                    {"series": [{"template": "json"}, {"template": "nodes"}]}
                ],
            }
        )

        assert psr._discover_stages(tmp_path, figures) == [
            "1_Initial",
            "2_Null",
            "3_Oude",
            "3_Zomerkade",
            "10_End",
        ]

    def test_raises_when_no_stage_files_match(self, tmp_path):
        with pytest.raises(ValueError, match="no files matched"):
            psr._discover_stages(
                tmp_path,
                [{"series": [{"file": "{stage}_output.json"}]}],
            )


kratos_profile_series = {
    "file": "{stage}__output.json",
    "node_set": "wall",
    "variable": "BENDING_MOMENT",
    "scale": 0.001,
    "label": "Kratos",
}


class TestRun:
    def test_auto_discovers_stages_from_series_files(self, case_dir):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "bending_moments.svg",
                    "type": "profiles_per_stage",
                    "series": [kratos_profile_series],
                }
            ],
            stages="auto",
        )

        psr.run(config_path)

        assert (case_dir / "plots" / "bending_moments.svg").is_file()

    def test_creates_one_file_per_figure(self, case_dir):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "bending_moments.svg",
                    "type": "profiles_per_stage",
                    "series": [kratos_profile_series],
                },
                {
                    "name": "history.svg",
                    "type": "histories_per_node",
                    "node_set": "wall",
                    "series": [
                        {"file": "{stage}__output.json", "variable": "BENDING_MOMENT"}
                    ],
                },
            ],
        )

        psr.run(config_path)

        assert (case_dir / "plots" / "bending_moments.svg").is_file()
        assert (case_dir / "plots" / "history.svg").is_file()

    def test_writes_one_file_per_stage_for_single_figures(self, case_dir):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "displacement_{stage}.svg",
                    "type": "single",
                    "per_stage_files": True,
                    "title": "{stage}",
                    "annotate_min_max": True,
                    "series": [kratos_profile_series],
                }
            ],
        )

        psr.run(config_path)

        assert (case_dir / "plots" / "displacement_stage_1.svg").is_file()
        assert (case_dir / "plots" / "displacement_stage_2.svg").is_file()

    def test_output_directory_can_be_overridden(self, case_dir, tmp_path):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "bending_moments.svg",
                    "type": "profiles_per_stage",
                    "series": [kratos_profile_series],
                }
            ],
        )
        elsewhere = tmp_path / "elsewhere"

        psr.run(config_path, elsewhere)

        assert (elsewhere / "bending_moments.svg").is_file()
        assert not (case_dir / "plots").exists()

    def test_node_sets_and_coordinates_may_be_given_explicitly(self, case_dir):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "bending_moments.svg",
                    "type": "profiles_per_stage",
                    "series": [kratos_profile_series],
                }
            ],
            mesh={"node_coordinates": {"2": [1.0, -5.0], "3": [2.0, -7.0]}},
            node_sets={"wall": {"ids": [2, 3]}},
        )

        pending = psr.run(config_path, show=True)

        first_stage_line = pending[0].figure.axes[0].lines[0]
        assert list(first_stage_line.get_xdata()) == [3.0, 2.0]
        assert list(first_stage_line.get_ydata()) == [-7.0, -5.0]

        psr.run(config_path)

        assert (case_dir / "plots" / "bending_moments.svg").is_file()

    def test_custom_subplots_use_their_own_series_and_axis_options(self, case_dir):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "custom.svg",
                    "type": "custom_subplots",
                    "subplots": [
                        {
                            "title": "First stage",
                            "xlabel": "Moment [kNm]",
                            "series": [
                                {**kratos_profile_series, "file": "stage_1__output.json"}
                            ],
                        },
                        {
                            "title": "Second stage",
                            "invert_yaxis": True,
                            "series": [
                                {**kratos_profile_series, "file": "stage_2__output.json"}
                            ],
                        },
                    ],
                }
            ],
        )

        pending = psr.run(config_path, show=True)

        assert len(pending) == 1
        first_axes, second_axes = pending[0].figure.axes
        assert first_axes.get_title() == "First stage"
        assert first_axes.get_xlabel() == "Moment [kNm]"
        assert list(first_axes.lines[0].get_xdata()) == [3.0, 2.0, 1.0]
        assert second_axes.get_title() == "Second stage"
        assert second_axes.yaxis_inverted()
        assert list(second_axes.lines[0].get_xdata()) == [6.0, 4.0, 0.0]

        psr.run(config_path)

        assert (case_dir / "plots" / "custom.svg").is_file()

    def test_figure_without_any_data_is_skipped(self, case_dir, capsys):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "empty.svg",
                    "type": "profiles_per_stage",
                    "stages": ["missing_stage"],
                    "series": [kratos_profile_series],
                }
            ],
        )

        psr.run(config_path)

        assert not (case_dir / "plots" / "empty.svg").exists()
        assert "no data for figure 'empty.svg'" in capsys.readouterr().out

    def test_unknown_figure_type_is_reported(self, case_dir):
        config_path = write_config(
            case_dir, [{"name": "x.svg", "type": "contour", "series": []}]
        )

        with pytest.raises(ValueError, match="contour"):
            psr.run(config_path)


two_stage_single_figures = [
    {
        "name": "displacement_{stage}.svg",
        "type": "single",
        "per_stage_files": True,
        "series": [kratos_profile_series],
    }
]


class TestShow:
    def test_show_collects_figures_without_writing_files(self, case_dir):
        config_path = write_config(case_dir, two_stage_single_figures)

        pending = psr.run(config_path, show=True)

        assert [item.path.name for item in pending] == [
            "displacement_stage_1.svg",
            "displacement_stage_2.svg",
        ]
        assert not (case_dir / "plots").exists()

    def test_viewer_switches_between_all_and_one_by_one(self, case_dir):
        pending = psr.run(write_config(case_dir, two_stage_single_figures), show=True)
        viewer = psr.FigureViewer(pending)

        assert len(viewer._content_axes) == 2

        viewer.show_one_by_one()
        assert len(viewer._content_axes) == 1
        assert viewer._content_axes[0].get_title().startswith("displacement_stage_1.svg")

        viewer.next()
        assert viewer._content_axes[0].get_title().startswith("displacement_stage_2.svg")

        viewer.next()
        assert viewer.index == 0
        viewer.previous()
        assert viewer.index == 1

        viewer.show_all()
        assert len(viewer._content_axes) == 2
        psr.plt.close(viewer.figure)

    def test_viewer_save_button_writes_all_figures(self, case_dir):
        pending = psr.run(write_config(case_dir, two_stage_single_figures), show=True)
        viewer = psr.FigureViewer(pending)

        viewer.save()

        assert (case_dir / "plots" / "displacement_stage_1.svg").is_file()
        assert (case_dir / "plots" / "displacement_stage_2.svg").is_file()
        psr.plt.close(viewer.figure)

    def test_main_with_show_opens_the_viewer(self, case_dir, monkeypatch):
        config_path = write_config(case_dir, two_stage_single_figures)
        monkeypatch.setattr(psr.plt, "switch_backend", lambda backend: None)
        shown = []
        monkeypatch.setattr(psr.plt, "show", lambda: shown.append(True))

        assert psr.main([str(config_path), "--show"]) == 0

        assert shown == [True]
        assert not (case_dir / "plots").exists()


class TestMain:
    def test_returns_zero_on_success(self, case_dir):
        config_path = write_config(
            case_dir,
            [
                {
                    "name": "bending_moments.svg",
                    "type": "profiles_per_stage",
                    "series": [kratos_profile_series],
                }
            ],
        )

        assert psr.main([str(config_path)]) == 0

    def test_returns_non_zero_and_a_message_on_a_faulty_config(self, case_dir, capsys):
        config_path = write_config(
            case_dir, [{"name": "x.svg", "type": "contour", "series": []}]
        )

        assert psr.main([str(config_path)]) == 1
        assert "Error:" in capsys.readouterr().err

    def test_prints_the_schema(self, capsys):
        assert psr.main(["--print-schema", "unused.json"]) == 0
        assert "figures" in capsys.readouterr().out
