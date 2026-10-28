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
