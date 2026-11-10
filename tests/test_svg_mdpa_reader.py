import pytest

from kratos_visualization_tools.svg import geometry, mdpa_reader


def test_parse_mdpa_maps_submodelpart_membership_and_quadratic_condition(mdpa_svg_input):
    model = mdpa_reader.parse_mdpa(mdpa_svg_input)

    assert model.points[5] == (0.5, 0)
    assert model.physical_surfaces == {"Soil": [1, 2]}
    assert set(model.physical_curves) == {"Bottom", "NodesOnly"}
    assert geometry._curve_points_by_name(model, ["Bottom"]) == [
        [(0, 0), (0.5, 0), (1, 0)],
    ]
    assert geometry._loop_to_polygon_points(model, 1) == [
        (0, 0), (1, 0), (0, 1), (0, 0),
    ]
    assert geometry._curve_points_by_name(model, ["NodesOnly"]) == [[(0, 0), (1, 0)]]


@pytest.mark.parametrize(
    ("entity_type", "node_ids", "expected_path", "is_surface"),
    [
        ("Element2D3N", "1 2 3", [1, 2, 3, 1], True),
        ("Element2D4N", "1 2 3 4", [1, 2, 3, 4, 1], True),
        ("SmallStrainUPwDiffOrderElement2D6N", "1 2 3 4 5 6", [1, 4, 2, 5, 3, 6, 1], True),
        (
            "SmallStrainUPwDiffOrderElement2D8N", "1 2 3 4 5 6 7 8",
            [1, 5, 2, 6, 3, 7, 4, 8, 1], True,
        ),
        ("LineElement2D2N", "1 2", [1, 2], False),
        ("LinearTimoshenkoBeamElement2D3N", "1 2 3", [1, 3, 2], False),
    ],
)
def test_parse_mdpa_reuses_geometry_paths_and_type_name_fallback(
    tmp_path, entity_type, node_ids, expected_path, is_surface
):
    source = tmp_path / "geometry.mdpa"
    source.write_text(
        "Begin Nodes\n"
        + "".join(f"{i} {i} {i % 2} 0\n" for i in range(1, 9))
        + f"End Nodes\nBegin Elements {entity_type}\n1 0 {node_ids}\nEnd Elements\n",
        encoding="utf-8",
    )

    model = mdpa_reader.parse_mdpa(source)

    assert model.splines[1] == expected_path
    groups = model.physical_surfaces if is_surface else model.physical_curves
    assert groups == {entity_type: [1]}


def test_parse_mdpa_preserves_multiple_memberships(mdpa_svg_input):
    with mdpa_svg_input.open("a", encoding="utf-8") as stream:
        stream.write(
            "Begin SubModelPart Other\nBegin SubModelPartElements\n1\n"
            "End SubModelPartElements\nEnd SubModelPart\n"
        )

    model = mdpa_reader.parse_mdpa(mdpa_svg_input)

    assert model.physical_surfaces == {"Other": [1], "Soil": [1, 2]}


def test_parse_mdpa_preserves_both_interface_faces(tmp_path):
    source = tmp_path / "interface.mdpa"
    source.write_text(
        "Begin Nodes\n"
        + "".join(f"{i} {i} 0 0\n" for i in range(1, 7))
        + "End Nodes\nBegin Elements Geo_UPwLineInterfacePlaneStrainDiffOrderElement3Plus3N\n"
        "1 0 1 2 3 4 5 6\nEnd Elements\n",
        encoding="utf-8",
    )

    model = mdpa_reader.parse_mdpa(source)

    assert model.splines == {1: [1, 3, 2], 2: [4, 6, 5]}
    assert list(model.physical_curves.values()) == [[1, 2]]
    assert not model.physical_surfaces


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "No nodes parsed from MDPA"),
        ("Begin Nodes\n1 0 0 0\nEnd Nodes\n", "No elements or conditions"),
        (
            "Begin Nodes\n1 0 0 0\nEnd Nodes\n"
            "Begin Elements Element2D3N\n1 0 1 2 3\nEnd Elements\n",
            "Elements 1 references missing nodes",
        ),
        (
            "Begin Nodes\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\nEnd Nodes\n"
            "Begin Elements Element3D4N\n1 0 1 2 3 4\nEnd Elements\n",
            "Elements 1: Unknown geometry",
        ),
    ],
)
def test_parse_mdpa_reports_missing_or_unsupported_geometry(tmp_path, text, message):
    source = tmp_path / "invalid.mdpa"
    source.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        mdpa_reader.parse_mdpa(source)


def test_parse_mdpa_names_nested_parts_and_draws_node_only_groups_on_outer_edges(tmp_path):
    source = tmp_path / "nested.mdpa"
    source.write_text(
        """Begin Nodes
1 0 0 0
2 1 0 0
3 0 1 0
4 1 1 0
End Nodes
Begin Elements Element2D3N
1 0 1 2 3
2 0 2 4 3
End Elements
Begin SubModelPart Soil // parent
  Begin SubModelPartData
  PRESSURE 1.0
  End SubModelPartData
  Begin SubModelPart Upper
    Begin SubModelPartNodes
    2
    3
    4
    End SubModelPartNodes
    Begin SubModelPartElements
    2
    End SubModelPartElements
  End SubModelPart
  Begin SubModelPartElements
  1
  End SubModelPartElements
End SubModelPart
Begin SubModelPart Diagonal
  Begin SubModelPartNodes
  2
  3
  End SubModelPartNodes
End SubModelPart
Begin SubModelPart Right
  Begin SubModelPartNodes
  2
  4
  End SubModelPartNodes
End SubModelPart
Begin SubModelPart AlsoRight
  Begin SubModelPartNodes
  2
  4
  End SubModelPartNodes
End SubModelPart
Begin SubModelPart Edges
  Begin SubModelPart Top
    Begin SubModelPartNodes
    3
    4
    End SubModelPartNodes
  End SubModelPart
End SubModelPart
""",
        encoding="utf-8",
    )

    entity_groups, node_groups = mdpa_reader._mdpa_memberships(source)
    model = mdpa_reader.parse_mdpa(source)

    assert entity_groups == {("Elements", 2): {"Soil/Upper"}, ("Elements", 1): {"Soil"}}
    assert node_groups["Edges/Top"] == {3, 4}
    assert model.physical_surfaces == {"Soil": [1], "Soil/Upper": [2]}
    assert set(model.physical_curves) == {"Right", "AlsoRight", "Edges/Top"}
    assert model.physical_curves["Right"] == model.physical_curves["AlsoRight"]
    assert geometry._curve_points_by_name(model, ["Right"]) == [[(1, 0), (1, 1)]]
    assert geometry._curve_points_by_name(model, ["Edges/Top"]) == [[(1, 1), (0, 1)]]
