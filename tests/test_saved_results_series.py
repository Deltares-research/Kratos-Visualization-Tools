import pytest

import plot_saved_results as psr


@pytest.fixture
def builder(case_dir, mesh_file):
    mesh = psr.read_mdpa(mesh_file)
    return psr.SeriesBuilder(case_dir, mesh, {"wall": mesh.node_ids("wall")})


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
