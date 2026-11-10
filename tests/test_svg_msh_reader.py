import pytest

from kratos_visualization_tools.svg import models, msh_reader


def test_parse_msh_v2_reads_physical_names_nodes_and_element_dimensions(tmp_path):
    path = tmp_path / "model.msh"
    path.write_text(
        """$PhysicalNames
2
1 5 "Boundary"
2 7 "Soil"
$EndPhysicalNames
$Nodes
3
1 0 0 0
2 1 0 0
3 0 1 0
$EndNodes
$Elements
2
1 1 2 5 0 1 2
2 2 2 7 0 1 2 3
$EndElements
""",
        encoding="utf-8",
    )

    model = msh_reader.parse_msh(path)

    assert model.physical_names == {5: (1, "Boundary"), 7: (2, "Soil")}
    assert model.nodes == {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0)}
    assert model.line_elements == [(1, 1, [1, 2], [5])]
    assert model.elements == [(2, 2, [1, 2, 3], [7])]


def test_parse_msh_v4_reads_entity_physical_tags_and_blocks(tmp_path):
    path = tmp_path / "model_v4.msh"
    path.write_text(
        """$PhysicalNames
2
1 5 "Boundary"
2 7 "Soil"
$EndPhysicalNames
$Entities
0 1 1 0
1 0 0 0 1 0 0 1 5 2 1 2
1 0 0 0 1 1 0 1 7 1 1
$EndEntities
$Nodes
1 3 1 3
2 1 0 3
1 2 3
0 0 0
1 0 0
0 1 0
$EndNodes
$Elements
2 2 1 2
1 1 1 1
1 1 2
2 1 2 1
2 1 2 3
$EndElements
""",
        encoding="utf-8",
    )

    model = msh_reader.parse_msh(path)

    assert model.curve_entity_phys == {1: [5]}
    assert model.surface_entity_phys == {1: [7]}
    assert model.line_elements == [(1, 1, [1, 2], [5])]
    assert model.elements == [(2, 2, [1, 2, 3], [7])]


@pytest.mark.parametrize("element_type", [3, 10, 16, 36, 37, 38, 39, 40, 41])
def test_mesh_corner_count_recognizes_quadrilaterals(element_type):
    assert msh_reader._mesh_corner_count(element_type) == 4


@pytest.mark.parametrize("element_type", [0, 2, 4, 15, 999])
def test_line_corner_count_rejects_non_line_elements(element_type):
    assert msh_reader._line_corner_count(element_type) == 0


@pytest.mark.parametrize("section", ["PhysicalNames", "Entities", "Nodes", "Elements"])
def test_parse_msh_handles_section_without_header(tmp_path, section):
    path = tmp_path / "partial.msh"
    path.write_text(f"${section}\n", encoding="utf-8")

    assert msh_reader.parse_msh(path) == models.MshModel()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("$Entities\n0 0\n", r"Invalid \$Entities header"),
        ("$Nodes\n1 1 1 1\n2 1\n", r"Invalid \$Nodes block header"),
        (
            "$Nodes\n1 1 1 1\n2 1 1 1\n",
            "Parametric nodes.*not supported",
        ),
        ("$Elements\n1 1 1 1\n2 1\n", r"Invalid \$Elements block header"),
    ],
)
def test_parse_msh_reports_invalid_block_headers(tmp_path, text, message):
    path = tmp_path / "invalid.msh"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        msh_reader.parse_msh(path)


def test_parse_msh_skips_unused_entities_and_section_trailers(tmp_path):
    path = tmp_path / "sections.msh"
    path.write_text(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        '$PhysicalNames\n1\n2 7 "Soil"\nignored\n$EndPhysicalNames\n'
        "$Entities\n1 0 0 1\n1 0 0 0 0\n1 0 0 0 1 1 1 0 0\n"
        "ignored\n$EndEntities\n"
        "$Nodes\n1\n1 0 0 0\nignored\n$EndNodes\n"
        "$Elements\n1\n1 15 0 1\nignored\n$EndElements\n",
        encoding="utf-8",
    )

    model = msh_reader.parse_msh(path)

    assert model.physical_names == {7: (2, "Soil")}
    assert model.nodes == {1: (0, 0, 0)}
    assert model.elements == model.line_elements == []


def test_parse_msh_v4_skips_element_trailer(tmp_path):
    path = tmp_path / "elements.msh"
    path.write_text(
        "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\nignored\n$EndElements\n",
        encoding="utf-8",
    )

    assert msh_reader.parse_msh(path).elements == [(1, 2, [1, 2, 3], [])]
