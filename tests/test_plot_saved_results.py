import json

import pytest

import plot_saved_results as psr


def test_legacy_table_helper_patch_reaches_package_reader(monkeypatch, tmp_path):
    from kratos_visualization_tools import saved_results

    path = tmp_path / "data.csv"
    path.write_text("1,2\n")
    original = psr._looks_like_header
    with monkeypatch.context() as patch:
        patch.setattr(psr, "_looks_like_header", lambda row: True)
        assert saved_results.read_table(path).header == ["1", "2"]
    assert saved_results._looks_like_header is original


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


@pytest.fixture
def mesh_file(tmp_path):
    path = tmp_path / "model.mdpa"
    path.write_text(
        "Begin ModelPartData\nEnd ModelPartData\n"
        "Begin Nodes\n"
        "  1 0.0  0.0 0.0\n"
        "  2 1.0 -1.0 0.0\n"
        "  3 2.0 -2.0 0.0\n"
        "End Nodes\n"
        "Begin Elements UPwSmallStrainElement2D3N\n"
        "  1 0 1 2 3\n"
        "End Elements\n"
        "Begin SubModelPart wall\n"
        "  Begin SubModelPartNodes\n"
        "    1\n"
        "    2\n"
        "    3\n"
        "  End SubModelPartNodes\n"
        "  Begin SubModelPart wall_tip\n"
        "    Begin SubModelPartNodes\n"
        "      3\n"
        "    End SubModelPartNodes\n"
        "  End SubModelPart\n"
        "End SubModelPart\n"
    )
    return path


def write_stage_output(directory, stage, time, values_by_node_id):
    data = {"TIME": [time]}
    for node_id, value in values_by_node_id.items():
        data[f"NODE_{node_id}"] = {
            "BENDING_MOMENT": [value],
            "TRACTION": [[value, 2.0 * value]],
        }
    (directory / f"{stage}__output.json").write_text(json.dumps(data))


@pytest.fixture
def case_dir(tmp_path, mesh_file):
    write_stage_output(tmp_path, "stage_1", 0.0, {1: 1000.0, 2: 2000.0, 3: 3000.0})
    write_stage_output(tmp_path, "stage_2", 1.0, {1: 0.0, 2: 4000.0, 3: 6000.0})
    return tmp_path


@pytest.fixture
def builder(case_dir, mesh_file):
    mesh = psr.read_mdpa(mesh_file)
    return psr.SeriesBuilder(case_dir, mesh, {"wall": mesh.node_ids("wall")})


class TestReadMdpa:
    def test_reads_nodal_coordinates(self, mesh_file):
        mesh = psr.read_mdpa(mesh_file)

        assert mesh.coordinates == {
            1: (0.0, 0.0, 0.0),
            2: (1.0, -1.0, 0.0),
            3: (2.0, -2.0, 0.0),
        }

    def test_reads_nested_sub_model_part_nodes(self, mesh_file):
        mesh = psr.read_mdpa(mesh_file)

        assert mesh.node_ids("wall") == [1, 2, 3]
        assert mesh.node_ids("wall_tip") == [3]

    def test_accepts_fully_qualified_sub_model_part_name(self, mesh_file):
        mesh = psr.read_mdpa(mesh_file)

        assert mesh.node_ids("PorousDomain.wall") == [1, 2, 3]

    def test_raises_for_unknown_sub_model_part(self, mesh_file):
        mesh = psr.read_mdpa(mesh_file)

        with pytest.raises(KeyError, match="does_not_exist"):
            mesh.node_ids("does_not_exist")


class TestReadTable:
    def test_reads_comma_separated_file_with_header(self, tmp_path):
        path = tmp_path / "with_header.csv"
        path.write_text("Level [m], Moment [kNm/m]\n6.2, 1.5\n5.0, 2.5\n")

        table = psr.read_table(path)

        assert table.header == ["Level [m]", "Moment [kNm/m]"]
        assert table.column("Moment [kNm/m]") == ["1.5", "2.5"]

    def test_reads_whitespace_separated_file_without_header(self, tmp_path):
        path = tmp_path / "no_header.csv"
        path.write_text("# Level  Moment\n# [m]    [kNm]\n0.0  1.0\n-1.0  2.0\n")

        table = psr.read_table(path)

        assert table.header is None
        assert table.column(0) == ["0.0", "-1.0"]
        assert table.column(1) == ["1.0", "2.0"]

    def test_strips_byte_order_mark(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_bytes("\ufeffLevel,Moment\n1.0,2.0\n".encode("utf-8-sig"))

        table = psr.read_table(path)

        assert table.header == ["Level", "Moment"]

    def test_column_by_name_requires_a_header(self, tmp_path):
        path = tmp_path / "no_header.csv"
        path.write_text("0.0,1.0\n")

        with pytest.raises(KeyError, match="no header row"):
            psr.read_table(path).column("Level")

    def test_has_column_reports_missing_columns(self, tmp_path):
        path = tmp_path / "with_header.csv"
        path.write_text("Level,Moment\n1.0,2.0\n")

        table = psr.read_table(path)

        assert table.has_column("Moment")
        assert not table.has_column("Shear")
        assert not table.has_column(5)


class TestKratosJsonSeries:
    def test_profile_is_sorted_by_y_coordinate_and_scaled(self, builder):
        spec = {
            "file": "{stage}__output.json",
            "node_set": "wall",
            "variable": "BENDING_MOMENT",
            "scale": 0.001,
            "label": "Kratos",
        }
        context = {"stage": "stage_1", "node": None}

        series = builder.build(spec, context)

        assert series.label == "Kratos"
        assert series.data_points == [(3.0, -2.0), (2.0, -1.0), (1.0, 0.0)]

        spec_with_node_ids = {**spec, "node_ids": ["1", "3"]}
        del spec_with_node_ids["node_set"]

        assert builder.build(spec_with_node_ids, context).data_points == [
            (3.0, -2.0),
            (1.0, 0.0),
        ]

    def test_profile_can_plot_a_vector_component_against_the_x_coordinate(self, builder):
        series = builder.build(
            {
                "file": "{stage}__output.json",
                "node_set": "wall",
                "variable": "TRACTION",
                "component": 1,
                "scale": 0.001,
                "x": "node_x",
                "y": "value",
                "sort_by": "x",
            },
            {"stage": "stage_1", "node": None},
        )

        assert series.data_points == [(0.0, 2.0), (1.0, 4.0), (2.0, 6.0)]

    def test_history_collects_one_point_per_stage(self, builder):
        series = builder.build(
            {
                "file": "{stage}__output.json",
                "variable": "BENDING_MOMENT",
                "scale": 0.001,
            },
            {"stages": ["stage_1", "stage_2"], "node": 2, "stage": None},
        )

        assert series.data_points == [(0.0, 2.0), (1.0, 4.0)]

    def test_history_skips_values_of_not_yet_activated_nodes(self, builder):
        context = {"stages": ["stage_1", "stage_2"], "node": 1, "stage": None}
        spec = {"file": "{stage}__output.json", "variable": "BENDING_MOMENT"}

        assert builder.build(spec, context).data_points == [(0.0, 1000.0)]
        assert builder.build({**spec, "skip_zero_values": False}, context).data_points == [
            (0.0, 1000.0),
            (1.0, 0.0),
        ]

    def test_missing_file_yields_no_series(self, builder, capsys):
        series = builder.build(
            {
                "file": "{stage}__output.json",
                "node_set": "wall",
                "variable": "BENDING_MOMENT",
            },
            {"stage": "no_such_stage", "node": None},
        )

        assert series is None
        assert "does not exist" in capsys.readouterr().out

    def test_unknown_node_set_is_reported(self, builder):
        with pytest.raises(KeyError, match="unknown_set"):
            builder.build(
                {
                    "file": "{stage}__output.json",
                    "node_set": "unknown_set",
                    "variable": "BENDING_MOMENT",
                },
                {"stage": "stage_1", "node": None},
            )

    def test_unknown_source_is_reported(self, builder):
        with pytest.raises(ValueError, match="gid_output"):
            builder.build({"source": "gid_output"}, {"stage": "stage_1", "node": None})


class TestCsvSeries:
    def test_columns_are_selected_by_name_and_transformed(self, builder, case_dir):
        (case_dir / "stage_1__comparison.csv").write_text(
            "Level [m],Shear Force [kN/m]\n0.0,1.5\n-1.0,2.5\n"
        )

        series = builder.build(
            {
                "source": "csv",
                "file": "{stage}__comparison.csv",
                "x_column": "Shear Force [kN/m]",
                "y_column": "Level [m]",
                "x_scale": -1.0,
            },
            {"stage": "stage_1", "node": None},
        )

        assert series.data_points == [(-1.5, 0.0), (-2.5, -1.0)]

    def test_column_names_may_contain_the_node_placeholder(self, builder, case_dir):
        (case_dir / "history.csv").write_text(
            "Time [s],node_2 - Ux [m]\n0.0,0.1\n1.0,0.2\n"
        )

        series = builder.build(
            {
                "source": "csv",
                "file": "history.csv",
                "x_column": "Time [s]",
                "y_column": "node_{node} - Ux [m]",
            },
            {"node": 2, "stage": None},
        )

        assert series.data_points == [(0.0, 0.1), (1.0, 0.2)]

    def test_rows_with_missing_values_are_skipped(self, builder, case_dir):
        (case_dir / "history.csv").write_text(
            "Time [s],Ux [m]\n0.0,N/A\n1.0,0.2\n"
        )

        series = builder.build(
            {
                "source": "csv",
                "file": "history.csv",
                "x_column": "Time [s]",
                "y_column": "Ux [m]",
            },
            {"node": None, "stage": None},
        )

        assert series.data_points == [(1.0, 0.2)]

    def test_unknown_column_yields_no_series(self, builder, case_dir, capsys):
        (case_dir / "history.csv").write_text("Time [s],Ux [m]\n0.0,0.1\n")

        series = builder.build(
            {
                "source": "csv",
                "file": "history.csv",
                "x_column": "Time [s]",
                "y_column": "Uy [m]",
            },
            {"node": None, "stage": None},
        )

        assert series is None
        assert "not found" in capsys.readouterr().out

    def test_rows_can_be_filtered_by_column_value(self, builder, case_dir):
        (case_dir / "line_extract.csv").write_text(
            "file_base,s,value\n"
            "stage_1,0.0,10.0\n"
            "stage_2,0.0,20.0\n"
            "stage_1,1.0,11.0\n"
            "stage_2,1.0,21.0\n"
        )
        spec = {
            "source": "csv",
            "file": "line_extract.csv",
            "x_column": "value",
            "y_column": "s",
            "filter": {"file_base": "{stage}"},
        }

        series = builder.build(spec, {"stage": "stage_2", "node": None})

        assert series.data_points == [(20.0, 0.0), (21.0, 1.0)]

    def test_series_can_expand_over_stages(self, builder, case_dir):
        (case_dir / "line_extract.csv").write_text(
            "file_base,s,value\n"
            "stage_1,-1.0,10.0\n"
            "stage_1,-2.0,11.0\n"
            "stage_2,-1.0,20.0\n"
            "stage_2,-2.0,21.0\n"
        )
        collection = psr._collect(
            builder,
            [
                {
                    "source": "csv",
                    "file": "line_extract.csv",
                    "label": "{stage}",
                    "x_column": "value",
                    "y_column": "s",
                    "sort_by": "y",
                    "filter": {"file_base": "{stage}"},
                    "expand_over_stages": True,
                }
            ],
            {"stages": ["stage_1", "stage_2"], "stage": None, "node": None},
            expand_over_stages=True,
        )

        assert [series.label for series in collection] == ["stage_1", "stage_2"]
        assert [series.data_points for series in collection] == [
            [(11.0, -2.0), (10.0, -1.0)],
            [(21.0, -2.0), (20.0, -1.0)],
        ]

    def test_unknown_filter_column_yields_no_series(self, builder, case_dir, capsys):
        (case_dir / "line_extract.csv").write_text("s,value\n0.0,10.0\n")

        series = builder.build(
            {
                "source": "csv",
                "file": "line_extract.csv",
                "x_column": "value",
                "y_column": "s",
                "filter": {"file_base": "stage_1"},
            },
            {"node": None, "stage": None},
        )

        assert series is None
        assert "filter column 'file_base' not found" in capsys.readouterr().out


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
