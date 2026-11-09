import json
import runpy
import sys
import xml.etree.ElementTree as ET

import pytest

import geo_to_svg as converter


SVG = f"{{{converter.SVG_NS}}}"


@pytest.mark.parametrize(
    ("expression", "names", "expected"),
    [
        (
            "2 * length + Pi / 2",
            {"length": 3, "Pi": 3.141592653589793},
            6 + 3.141592653589793 / 2,
        ),
        ("-(a - 2) ** 2", {"a": 5}, -9),
        ("+4 / 2", {}, 2),
    ],
)
def test_eval_expr_supports_arithmetic_and_variables(expression, names, expected):
    assert converter.eval_expr(expression, names) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["missing + 1", "__import__('os')", "a[0]", "a // 2", "'text'"],
)
def test_eval_expr_rejects_unknown_or_unsupported_expressions(expression):
    with pytest.raises((ValueError, SyntaxError)):
        converter.eval_expr(expression, {"a": 2})


def test_parse_geo_reads_entities_variables_and_physical_groups(tmp_path):
    path = tmp_path / "model.geo"
    path.write_text(
        """
        // A square, with an inline comment.
        side = 2;
        Point(1) = {0, 0, 0, 1};
        Point(2) = {side, 0, 0, 1}; // lower right
        Point(3) = {side, side, 0, 1};
        Point(4) = {0, side, 0, 1};
        Line(1) = {1, 2};
        Line(2) = {2, 3};
        Spline(3) = {3, 4};
        Line(4) = {4, 1};
        Curve Loop(1) = {1, 2, 3, 4};
        Plane Surface(1) = {1};
        Physical Curve("Boundary", 8) = {1, -2};
        Physical Surface("Soil", 9) = {1};
        """,
        encoding="utf-8",
    )

    model = converter.parse_geo(path)

    assert model.vars["side"] == 2
    assert model.points[2] == (2, 0)
    assert model.lines[1] == (1, 2)
    assert model.splines[3] == [3, 4]
    assert model.curve_loops[1] == [1, 2, 3, 4]
    assert model.plane_surfaces[1] == [1]
    assert model.physical_curves["Boundary"] == [1, -2]
    assert model.physical_surfaces["Soil"] == [1]


def test_loop_polygon_follows_curve_direction_and_closes():
    model = converter.GeoModel(
        points={1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)},
        lines={1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)},
        curve_loops={1: [1, 2, 3, 4]},
    )

    assert converter._loop_to_polygon_points(model, 1) == [
        (0, 0), (1, 0), (1, 1), (0, 1), (0, 0),
    ]
    assert converter._entity_polyline(model, -1) == [(1, 0), (0, 0)]


@pytest.fixture
def curve_model():
    return converter.GeoModel(
        points={1: (0, 0), 2: (2, 0), 3: (2, 2), 4: (0, 2)},
        lines={
            1: (1, 2),
            2: (3, 2),
            3: (3, 4),
            4: (4, 1),
            5: (1, 99),
            6: (98, 99),
        },
        splines={7: [1, 99, 2, 3], 8: [98, 99]},
        physical_curves={
            "bottom": [1],
            "joined": [1, -2],
            "empty": [],
            "invalid": [5, 6, 99],
        },
    )


@pytest.mark.parametrize(
    ("signed_id", "expected"),
    [
        (1, [(0, 0), (2, 0)]),
        (-1, [(2, 0), (0, 0)]),
        (5, [(0, 0)]),
        (-5, [(0, 0)]),
        (6, []),
        (7, [(0, 0), (2, 0), (2, 2)]),
        (-7, [(2, 2), (2, 0), (0, 0)]),
        (8, []),
        (99, []),
        (0, []),
    ],
)
def test_entity_polyline_handles_direction_and_missing_points(
    curve_model, signed_id, expected
):
    assert converter._entity_polyline(curve_model, signed_id) == expected


@pytest.mark.parametrize(
    ("curve_ids", "expected"),
    [
        ([], []),
        ([99, 6], []),
        ([1], [(0, 0), (2, 0)]),
        ([1, 2, 3], [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
        ([1, -2, 3, 4], [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
        ([1, 99, 3], [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
    ],
)
def test_loop_polygon_joins_reverses_skips_and_closes(curve_model, curve_ids, expected):
    curve_model.curve_loops[1] = curve_ids

    assert converter._loop_to_polygon_points(curve_model, 1) == expected


def test_loop_polygon_returns_empty_for_unknown_loop(curve_model):
    assert converter._loop_to_polygon_points(curve_model, 99) == []


@pytest.mark.parametrize(
    ("curve_ids", "expected"),
    [
        ([], []),
        ([99, 5, 6], []),
        ([1, -2], [(0, 0), (2, 0), (2, 2)]),
        ([1, 5, 99, -2], [(0, 0), (2, 0), (2, 2)]),
        ([1, 3], [(0, 0), (2, 0), (2, 2), (0, 2)]),
        ([-7, -1], [(2, 2), (2, 0), (0, 0), (2, 0), (0, 0)]),
    ],
)
def test_collect_curve_points_handles_joins_and_incomplete_curves(
    curve_model, curve_ids, expected
):
    assert converter._collect_curve_points(curve_model, curve_ids) == expected


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        ([], []),
        (["missing", "empty", "invalid"], []),
        (
            ["joined", "missing", "empty", "invalid", "bottom"],
            [[(0, 0), (2, 0), (2, 2)], [(0, 0), (2, 0)]],
        ),
    ],
)
def test_curve_points_by_name_keeps_valid_groups_in_requested_order(
    curve_model, names, expected
):
    assert converter._curve_points_by_name(curve_model, names) == expected


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [(0, 0, 0), (3, 4, 5), (-3, -4, 5), (0, -2, 2), (1e200, 0, 1e200)],
)
def test_line_len_returns_euclidean_length(dx, dy, expected):
    assert converter._line_len(dx, dy) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("poly", "spacing", "expected"),
    [
        ([], 1, []),
        ([(1, 2)], 1, []),
        ([(0, 0), (0, 0), (0, 0)], 1, []),
        ([(0, 0), (1e-13, 0)], 1, []),
        ([(0, 0), (1, 0)], 2, []),
        (
            [(0, 0), (5, 0)],
            2,
            [(2, 0, 1, 0), (4, 0, 1, 0)],
        ),
        (
            [(0, 0), (2, 0), (2, 3)],
            2,
            [(2, 0, 1, 0), (2, 2, 0, 1)],
        ),
        (
            [(0, 0), (1, 0), (1, 3)],
            2,
            [(1, 1, 0, 1), (1, 3, 0, 1)],
        ),
        (
            [(0, 0), (0, 0), (2, 0), (2, 0), (2, 2)],
            2,
            [(2, 0, 1, 0), (2, 2, 0, 1)],
        ),
        (
            [(3, 4), (0, 0)],
            2.5,
            [(1.5, 2, -0.6, -0.8), (0, 0, -0.6, -0.8)],
        ),
        (
            [(0, 0), (1, 0)],
            0.25,
            [(0.25, 0, 1, 0), (0.5, 0, 1, 0), (0.75, 0, 1, 0), (1, 0, 1, 0)],
        ),
    ],
)
def test_sample_polyline_points_returns_positions_and_unit_tangents(
    poly, spacing, expected
):
    samples = converter._sample_polyline_points(poly, spacing)

    assert len(samples) == len(expected)
    for sample, expected_sample in zip(samples, expected):
        assert sample == pytest.approx(expected_sample)


@pytest.mark.parametrize("side_sign", [-1.0, 1.0])
def test_collect_vertical_side_samples_filters_segments_and_sorts_by_height(side_sign):
    polylines = [
        [],
        [(1, 2)],
        [(0, 0), (0, 0), (0, 1e-10)],
        [(0, 0), (10, 0)],
        [(0, 0), (10, 10)],
        [(5, 40), (5, 20)],
        [(0, 0), (0, 10), (3, 10)],
        [(0, 10), (3, 20)],
    ]

    samples = converter._collect_vertical_side_samples(polylines, side_sign)

    assert len(samples) == 3
    for sample, expected in zip(
        samples,
        [(0, 5, 0, 1), (1.5, 15, 3 / 109**0.5, 10 / 109**0.5), (5, 30, 0, -1)],
    ):
        assert sample == pytest.approx(expected)


@pytest.mark.parametrize(
    ("count", "indices"),
    [(0, []), (1, [0]), (2, [0, 1]), (3, [0, 2]), (5, [1, 3]), (9, [2, 6])],
)
def test_pick_two_samples_selects_quartiles_without_mutating_input(count, indices):
    candidates = [(0, i, 0, 1) for i in range(count)]
    original = candidates.copy()

    assert converter._pick_two_samples(candidates) == [candidates[i] for i in indices]
    assert candidates == original


@pytest.mark.parametrize("polylines", [[], [[]], [[(0, 0)]], [[(0, 0), (0, 0)]]])
@pytest.mark.parametrize("renderer", ["hatching", "fixity"])
def test_support_drawers_skip_empty_or_degenerate_geometry(polylines, renderer):
    group = ET.Element(f"{SVG}g")

    if renderer == "hatching":
        converter._draw_bottom_hatching(group, polylines, center_svg=(0, 10))
    else:
        converter._draw_side_fixity(group, polylines, outward_sign=-1)

    assert len(group) == 0


@pytest.mark.parametrize(
    ("poly", "center", "expected_dx", "expected_dy"),
    [
        ([(0, 0), (20, 0)], (10, 10), 5, -8.660254),
        ([(20, 0), (0, 0)], (10, 10), 5, -8.660254),
        ([(0, 0), (20, 0)], (10, -10), -5, 8.660254),
    ],
)
def test_draw_bottom_hatching_points_away_from_domain(
    poly, center, expected_dx, expected_dy
):
    group = ET.Element(f"{SVG}g")

    converter._draw_bottom_hatching(
        group, [poly], center_svg=center, spacing=10, hatch_len=10
    )

    lines = group.findall(f"{SVG}line")
    assert len(lines) == 2
    expected_xs = [10, 20] if poly[0] == (0, 0) else [10, 0]
    for line, expected_x in zip(lines, expected_xs):
        x1, y1, x2, y2 = (float(line.attrib[key]) for key in ("x1", "y1", "x2", "y2"))
        assert (x1, y1) == pytest.approx((expected_x, 0))
        assert (x2 - x1, y2 - y1) == pytest.approx(
            (expected_dx, expected_dy), abs=0.001
        )
        assert line.attrib["stroke"] == "#222"


@pytest.mark.parametrize("outward_sign", [-1.0, 1.0])
def test_draw_side_fixity_places_triangles_wheels_and_hatches_outside(outward_sign):
    group = ET.Element(f"{SVG}g")

    converter._draw_side_fixity(
        group, [[(100, 0), (100, 40)]], outward_sign=outward_sign
    )

    triangles = group.findall(f"{SVG}polygon")
    circles = group.findall(f"{SVG}circle")
    lines = group.findall(f"{SVG}line")
    assert len(triangles) == 1
    assert len(circles) == 2
    assert len(lines) == 4
    points = [
        tuple(float(value) for value in point.split(","))
        for point in triangles[0].attrib["points"].split()
    ]
    base_x = 100 + outward_sign * 12 * 3**0.5
    assert points[0] == (100, 20)
    assert points[1] == pytest.approx((base_x, 8), abs=0.001)
    assert points[2] == pytest.approx((base_x, 32), abs=0.001)
    for circle, y in zip(circles, [14, 26]):
        assert float(circle.attrib["cx"]) == pytest.approx(
            base_x + outward_sign * 3.2, abs=0.001
        )
        assert float(circle.attrib["cy"]) == y
        assert float(circle.attrib["r"]) == 3.2
    wall_x = base_x + outward_sign * 6.4
    wall = lines[0]
    assert float(wall.attrib["x1"]) == pytest.approx(wall_x, abs=0.001)
    assert wall.attrib["x1"] == wall.attrib["x2"]
    assert (float(wall.attrib["y1"]), float(wall.attrib["y2"])) == (8, 32)
    for hatch, y in zip(lines[1:], [10.4, 20, 29.6]):
        assert float(hatch.attrib["x1"]) == pytest.approx(wall_x, abs=0.001)
        assert float(hatch.attrib["x2"]) == pytest.approx(
            wall_x + outward_sign * 6, abs=0.001
        )
        assert float(hatch.attrib["y1"]) == pytest.approx(y)
        assert float(hatch.attrib["y2"]) == pytest.approx(y + 3)


def test_draw_side_fixity_limits_supports_to_two_and_ignores_horizontal_segments():
    group = ET.Element(f"{SVG}g")

    converter._draw_side_fixity(
        group,
        [[(0, 80), (0, 60), (0, 40), (0, 20), (0, 0)], [(0, 0), (50, 0)]],
        outward_sign=1,
    )

    triangles = group.findall(f"{SVG}polygon")
    assert len(triangles) == 2
    assert [triangle.attrib["points"].split()[0] for triangle in triangles] == [
        "0.000,30.000", "0.000,50.000",
    ]


@pytest.mark.parametrize("polylines", [[], [[]], [[(1, 2)]]])
def test_draw_load_arrows_skips_incomplete_polylines(polylines):
    group = ET.Element(f"{SVG}g")

    converter._draw_load_arrows(group, polylines)

    assert len(group) == 0


@pytest.mark.parametrize(
    ("poly", "expected_positions"),
    [
        ([(0, 20), (20, 20)], [(0, 20), (20, 20)]),
        ([(0, 20), (21, 20)], [(0, 20), (10.5, 20), (21, 20)]),
        ([(0, 20), (30, 20)], [(0, 20), (10, 20), (20, 20), (30, 20)]),
        ([(30, 20), (0, 20)], [(30, 20), (20, 20), (10, 20), (0, 20)]),
        ([(0, 0), (18, 24)], [(0, 0), (6, 8), (12, 16), (18, 24)]),
        ([(0, 20), (5, 20), (10, 20)], [(0, 20), (10, 20)]),
        (
            [(0, 20), (30, 20), (60, 20)],
            [(0, 20), (10, 20), (20, 20), (40, 20), (50, 20), (60, 20)],
        ),
    ],
)
def test_draw_load_arrows_places_shafts_heads_and_top_line(poly, expected_positions):
    group = ET.Element(f"{SVG}g")

    converter._draw_load_arrows(group, [poly], spacing=10, arrow_len=8)

    lines = group.findall(f"{SVG}line")
    heads = group.findall(f"{SVG}polygon")
    assert len(lines) == len(expected_positions) + 1
    assert len(heads) == len(expected_positions)
    for shaft, head, (x, y) in zip(lines[:-1], heads, expected_positions):
        assert tuple(
            float(shaft.attrib[key]) for key in ("x1", "y1", "x2", "y2")
        ) == pytest.approx((x, y - 8, x, y))
        head_points = [
            tuple(float(value) for value in point.split(","))
            for point in head.attrib["points"].split()
        ]
        assert head_points == [(x, y), (x - 4, y - 6), (x + 4, y - 6)]
        assert head.attrib["fill"] == shaft.attrib["stroke"] == "#1f1f1f"
    top = lines[-1]
    top_y = min(y for x, y in expected_positions) - 8
    assert tuple(
        float(top.attrib[key]) for key in ("x1", "y1", "x2", "y2")
    ) == pytest.approx(
        (
            min(x for x, y in expected_positions),
            top_y,
            max(x for x, y in expected_positions),
            top_y,
        )
    )


def test_draw_load_arrows_keeps_separate_load_curves_separate():
    group = ET.Element(f"{SVG}g")

    converter._draw_load_arrows(
        group, [[(0, 20), (10, 20)], [(30, 40), (40, 40)]]
    )

    assert len(group.findall(f"{SVG}polygon")) == 4
    lines = group.findall(f"{SVG}line")
    assert len(lines) == 6
    assert (lines[2].attrib["x1"], lines[2].attrib["x2"]) == ("0.000", "10.000")
    assert (lines[5].attrib["x1"], lines[5].attrib["x2"]) == ("30.000", "40.000")


@pytest.mark.parametrize("kind", ["curve", "surface"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Soil ", (["Soil"], "Soil")),
        ("  ", None),
        (None, None),
        (42, None),
        (["Soil"], None),
        ({}, None),
        ({"name": "Soil", "show": False}, None),
        ({"name": "  "}, None),
        ({"name": 42}, None),
        ({"name": " Soil "}, (["Soil"], "Soil")),
        ({"name": "Soil", "label": " Ground "}, (["Soil"], "Ground")),
        ({"name": "Soil", "label": ""}, (["Soil"], "Soil")),
        ({"name": "Soil", "label": " "}, (["Soil"], "Soil")),
        ({"name": "Soil", "label": 7}, (["Soil"], "7")),
        ({"names": []}, None),
        ({"names": [" ", ""]}, None),
        (
            {"names": [" Clay ", "", " Sand "], "label": " Ground "},
            (["Clay", "Sand"], "Ground"),
        ),
        ({"names": ["Clay", "Sand"]}, (["Clay", "Sand"], "Clay, Sand")),
        (
            {"name": "ignored", "names": ["Clay", "Sand"], "label": " "},
            (["Clay", "Sand"], "Clay"),
        ),
    ],
)
def test_parse_legend_entry_normalizes_filters_and_groups_names(raw, expected, kind):
    assert converter._parse_legend_entry(raw, kind) == expected


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

    model = converter.parse_msh(path)

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

    model = converter.parse_msh(path)

    assert model.curve_entity_phys == {1: [5]}
    assert model.surface_entity_phys == {1: [7]}
    assert model.line_elements == [(1, 1, [1, 2], [5])]
    assert model.elements == [(2, 2, [1, 2, 3], [7])]


def test_geo_to_svg_writes_geometry_and_configured_legend(tmp_path):
    model = converter.GeoModel(
        points={1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)},
        lines={1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)},
        curve_loops={1: [1, 2, 3, 4]},
        plane_surfaces={1: [1]},
        physical_curves={"Boundary": [1]},
        physical_surfaces={"Soil": [1]},
    )
    output = tmp_path / "nested" / "preview.svg"

    converter.geo_to_svg(
        model,
        output,
        width=400,
        height=300,
        padding=20,
        legend_cfg={
            "curves": [{"name": "Boundary", "label": "Model boundary"}],
            "phreatic_line": {"label": "Water level", "points": [[0, 0.5], [1, 0.5]]},
            "top_left": [12, 14],
        },
    )

    root = ET.parse(output).getroot()
    assert root.attrib["viewBox"] == "0 0 400 300"
    assert len(root.findall(f".//{SVG}g[@id='physical_surfaces']/{SVG}polygon")) == 2
    assert len(root.findall(f".//{SVG}g[@id='physical_curves']/{SVG}path")) == 1
    assert root.find(f".//{SVG}g[@id='legend']/{SVG}text").text == "Physical Curves (conditions)"
    assert any(node.text == "Model boundary" for node in root.findall(f".//{SVG}g[@id='legend']/{SVG}text"))
    assert any(node.text == "Water level" for node in root.findall(f".//{SVG}g[@id='legend']/{SVG}text"))
    assert root.find(f".//{SVG}g[@id='supports_and_loads']/{SVG}path") is not None


def test_msh_to_svg_draws_boundary_conditions_and_shared_edges(tmp_path):
    model = converter.MshModel(
        nodes={
            1: (0, 0, 0),
            2: (1, 0, 0),
            3: (0, 1, 0),
            4: (1, 1, 0),
        },
        physical_names={1: (2, "Left"), 2: (2, "Right"), 3: (1, "Boundary")},
        elements=[
            (1, 2, [1, 2, 3], [1]),
            (2, 2, [2, 4, 3], [2]),
        ],
        line_elements=[(3, 1, [1, 2], [3])],
    )
    output = tmp_path / "mesh.svg"

    converter.msh_to_svg(model, output, width=400, height=300, padding=20)

    root = ET.parse(output).getroot()
    elements = root.find(f".//{SVG}g[@id='mesh_elements']")
    shared = root.find(f".//{SVG}g[@id='mesh_shared_edges']")
    assert elements is not None
    assert shared is not None
    assert len(elements.findall(f"{SVG}path")) >= 1
    assert any(path.attrib["stroke"] == converter._hash_color("Boundary") for path in elements.findall(f"{SVG}path"))
    assert any(path.attrib["stroke"] == "#000000" for path in shared.findall(f"{SVG}path"))


@pytest.mark.parametrize(
    ("renderer", "model", "message"),
    [
        ("geo_to_svg", converter.GeoModel(), "No points parsed"),
        ("msh_to_svg", converter.MshModel(), "No nodes parsed"),
        ("msh_to_svg", converter.MshModel(nodes={1: (0, 0, 0)}), "No 2D mesh elements"),
    ],
)
def test_renderers_report_missing_geometry(tmp_path, renderer, model, message):
    with pytest.raises(ValueError, match=message):
        getattr(converter, renderer)(model, tmp_path / "invalid.svg", 100, 100, 5)


def test_eval_expr_rejects_unsupported_unary_operator():
    with pytest.raises(ValueError, match="Unsupported unary operator"):
        converter.eval_expr("~2", {})


def test_parse_geo_handles_unterminated_statements_and_invalid_definitions(tmp_path):
    path = tmp_path / "partial.geo"
    path.write_text(
        "invalid = unknown;\nPoint(9) = {1};\n"
        "Point(1) = {Pi, 2};\nPoint(2) = {0, 0}",
        encoding="utf-8",
    )

    model = converter.parse_geo(path)

    assert "invalid" not in model.vars
    assert model.points == {1: (model.vars["Pi"], 2), 2: (0, 0)}


@pytest.mark.parametrize("element_type", [3, 10, 16, 36, 37, 38, 39, 40, 41])
def test_mesh_corner_count_recognizes_quadrilaterals(element_type):
    assert converter._mesh_corner_count(element_type) == 4


@pytest.mark.parametrize("element_type", [0, 2, 4, 15, 999])
def test_line_corner_count_rejects_non_line_elements(element_type):
    assert converter._line_corner_count(element_type) == 0


@pytest.mark.parametrize("section", ["PhysicalNames", "Entities", "Nodes", "Elements"])
def test_parse_msh_handles_section_without_header(tmp_path, section):
    path = tmp_path / "partial.msh"
    path.write_text(f"${section}\n", encoding="utf-8")

    assert converter.parse_msh(path) == converter.MshModel()


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
        converter.parse_msh(path)


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

    model = converter.parse_msh(path)

    assert model.physical_names == {7: (2, "Soil")}
    assert model.nodes == {1: (0, 0, 0)}
    assert model.elements == model.line_elements == []


def test_parse_msh_v4_skips_element_trailer(tmp_path):
    path = tmp_path / "elements.msh"
    path.write_text(
        "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\nignored\n$EndElements\n",
        encoding="utf-8",
    )

    assert converter.parse_msh(path).elements == [(1, 2, [1, 2, 3], [])]


@pytest.mark.parametrize("points", [[], [(0, 0)]])
def test_draw_phreatic_line_skips_incomplete_geometry(points):
    group = ET.Element(f"{SVG}g")

    converter._draw_phreatic_line(group, points)

    assert len(group) == 0


def test_build_legend_spec_filters_entries_and_groups_surface_styles():
    model = converter.GeoModel(
        physical_curves={"Boundary": [1]},
        physical_surfaces={"Clay": [1], "Sand": [2]},
    )

    spec = converter._build_legend_spec(
        model,
        {
            "curves": [None, "missing", "Boundary"],
            "surfaces": [
                {"name": "Clay", "show": False},
                "missing",
                {"names": ["Clay", "missing", "Sand"], "label": "Ground"},
            ],
            "max_curves": 1,
            "max_surfaces": 2,
        },
    )

    assert [entry.names for entry in spec.curve_entries] == [["Boundary"]]
    assert [entry.names for entry in spec.surface_entries] == [["Clay", "Sand"]]
    assert spec.surface_entries[0].label == "Ground"
    assert spec.surface_style_by_name["Clay"] == spec.surface_style_by_name["Sand"]
    assert spec.surface_entries[0].style_key == spec.surface_style_by_name["Clay"]
    assert (spec.max_curve_items, spec.max_surface_items) == (1, 2)


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], "invalid"],
        [[0, 0], ["invalid", 1]],
        [[0, 0], [None, 1]],
    ],
)
def test_build_legend_spec_ignores_invalid_phreatic_points(points):
    spec = converter._build_legend_spec(
        converter.GeoModel(), {"phreatic_line": {"points": points}}
    )

    assert spec.phreatic_line_points is None


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (["invalid", 2], (20, 20)),
        ({"x": "12", "y": "34"}, (12, 34)),
        ({"x": 12}, (12, 20)),
        ({"x": None}, (20, 20)),
    ],
)
def test_build_legend_spec_parses_anchor_and_handles_invalid_values(anchor, expected):
    spec = converter._build_legend_spec(converter.GeoModel(), {"top_left": anchor})

    assert spec.legend_top_left == expected


def test_geo_to_svg_skips_missing_surfaces_short_loops_and_curves(tmp_path):
    model = converter.GeoModel(
        points={1: (0, 0), 2: (1, 1)},
        lines={1: (1, 2)},
        curve_loops={1: [1]},
        plane_surfaces={1: [1]},
        physical_surfaces={"Invalid": [99, 1]},
        physical_curves={"Invalid": [99]},
    )
    output = tmp_path / "partial.svg"

    converter.geo_to_svg(model, output, 100, 100, 5)

    root = ET.parse(output).getroot()
    assert len(root.find(f"{SVG}g[@id='physical_surfaces']")) == 0
    assert len(root.find(f"{SVG}g[@id='physical_curves']")) == 0


@pytest.fixture
def two_triangle_mesh():
    return converter.MshModel(
        nodes={1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (1, 1, 0)},
        physical_names={1: (2, "Clay"), 2: (2, "Sand")},
        elements=[(1, 2, [1, 2, 3], [1]), (2, 2, [2, 4, 3], [2])],
    )


@pytest.mark.parametrize("same_surface", [False, True])
def test_msh_to_svg_colors_shared_edges_by_surface_when_enabled(
    tmp_path, two_triangle_mesh, same_surface
):
    if same_surface:
        two_triangle_mesh.elements[1] = (2, 2, [2, 4, 3], [1])
    output = tmp_path / "shared.svg"

    converter.msh_to_svg(
        two_triangle_mesh, output, 100, 100, 5,
        legend_cfg={"msh": {"shared_edges_black": False, "boundary_edge_width": 2}},
    )

    root = ET.parse(output).getroot()
    paths = root.findall(f"{SVG}g[@id='mesh_shared_edges']/{SVG}path")
    assert len(paths) == 1
    allowed_colors = {converter._hash_color("Clay")}
    if not same_surface:
        allowed_colors.add(converter._hash_color("Sand"))
    assert paths[0].attrib["stroke"] in allowed_colors
    assert paths[0].attrib["stroke-width"] == "2.00"


def test_msh_to_svg_skips_invalid_elements_and_uses_unnamed_element_style(
    tmp_path, two_triangle_mesh
):
    two_triangle_mesh.elements = [
        (1, 2, [1, 2, 3], [999]),
        (2, 2, [1, 2], []),
        (3, 2, [1, 2, 99], []),
        (4, 4, [1, 2, 3, 4], []),
    ]
    two_triangle_mesh.line_elements = [
        (5, 1, [1], []),
        (6, 2, [1, 2, 3], []),
        (7, 1, [1, 2], [1, 999]),
    ]
    output = tmp_path / "fallback.svg"

    converter.msh_to_svg(
        two_triangle_mesh, output, 100, 100, 5, legend_cfg={"msh": []}
    )

    root = ET.parse(output).getroot()
    paths = root.findall(f"{SVG}g[@id='mesh_elements']/{SVG}path")
    assert len(paths) == 1
    assert paths[0].attrib["stroke"] == converter._hash_color("elem_1")
    assert any(node.text == "Elements rendered: 1 (line-only view)" for node in root.iter())


@pytest.mark.parametrize("nested", [False, True])
def test_script_entry_point_converts_geo_with_legend_config(
    tmp_path, monkeypatch, capsys, nested
):
    source = tmp_path / "model.GEO"
    source.write_text("Point(1) = {0, 0};", encoding="utf-8")
    output = tmp_path / "cli.svg"
    config = tmp_path / "legend.json"
    legend = {"top_left": [12, 34]}
    config.write_text(json.dumps({"legend": legend} if nested else legend), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", str(source), "--output", str(output),
        "--width", "300", "--height", "200", "--padding", "10",
        "--legend-config", str(config),
    ])

    runpy.run_path(converter.__file__, run_name="__main__")

    root = ET.parse(output).getroot()
    assert root.attrib["viewBox"] == "0 0 300 200"
    title = root.find(f"{SVG}g[@id='legend']/{SVG}text")
    assert (title.attrib["x"], title.attrib["y"]) == ("12.0", "34.0")
    stdout = capsys.readouterr().out
    assert f"Wrote SVG: {output}" in stdout
    assert "Parsed points=1, lines=0, splines=0, loops=0" in stdout
    assert "Physical curves=0, physical surfaces=0" in stdout


def test_main_converts_msh_using_defaults(tmp_path, monkeypatch, capsys):
    source = tmp_path / "model.msh"
    source.write_text(
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 0 1 2 3\n$EndElements\n",
        encoding="utf-8",
    )
    output = tmp_path / "mesh.svg"
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", str(source), "--output", str(output),
    ])

    converter.main()

    assert ET.parse(output).getroot().attrib["viewBox"] == "0 0 2200 1200"
    assert "Parsed nodes=3, elements=1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config_value", "message"),
    [
        ([], "Legend config root must be a JSON object"),
        ({"legend": []}, "Legend config must contain a JSON object"),
    ],
)
def test_main_rejects_non_object_legend_config(tmp_path, monkeypatch, config_value, message):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(config_value), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", "unused.geo", "--output", str(tmp_path / "out.svg"),
        "--legend-config", str(config),
    ])

    with pytest.raises(ValueError, match=message):
        converter.main()


def test_main_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", "model.txt", "--output", str(tmp_path / "out.svg"),
    ])

    with pytest.raises(ValueError, match="Unsupported input extension"):
        converter.main()


@pytest.mark.parametrize(
    ("bounds", "point", "expected"),
    [
        ((-2, 8, -5, 5), (-2, -5), (10, 190)),
        ((-2, 8, -5, 5), (8, 5), (290, 10)),
        ((-2, 8, -5, 5), (3, 0), (150, 100)),
        ((2, 2, 3, 3), (2, 3), (10, 190)),
    ],
)
def test_svg_transform_preserves_padding_y_inversion_and_degenerate_bounds(
    bounds, point, expected
):
    transform = converter._svg_transform(*bounds, width=300, height=200, padding=10)

    assert transform(point) == pytest.approx(expected)


def test_entity_polyline_prefers_line_when_spline_has_same_id(curve_model):
    curve_model.splines[1] = [3, 4]

    assert converter._entity_polyline(curve_model, 1) == [(0, 0), (2, 0)]


@pytest.mark.parametrize("kind", ["curve", "surface"])
def test_legend_entries_keep_group_indices_and_last_style_assignment(kind):
    entries, styles = converter._legend_entries(
        ["Clay", "Sand", "Other"],
        [
            None,
            {"names": ["Clay", "Sand", "missing"], "label": "Ground"},
            {"name": "Clay", "label": "Top"},
        ],
        kind,
    )

    assert entries == [
        converter.LegendEntry(["Clay", "Sand"], "Ground", f"{kind}_cfg_1_Ground"),
        converter.LegendEntry(["Clay"], "Top", f"{kind}_cfg_2_Top"),
    ]
    assert styles == {
        "Clay": f"{kind}_cfg_2_Top",
        "Sand": f"{kind}_cfg_1_Ground",
        "Other": "Other",
    }


@pytest.mark.parametrize("config", [None, {}, {"top_left": [20, 20]}])
def test_build_legend_spec_preserves_sorted_defaults(config):
    model = converter.GeoModel(
        physical_curves={"Z": [], "A": []},
        physical_surfaces={"Sand": [], "Clay": []},
    )

    spec = converter._build_legend_spec(model, config)

    assert [entry.label for entry in spec.curve_entries] == ["A", "Z"]
    assert [entry.label for entry in spec.surface_entries] == ["Clay", "Sand"]
    assert spec.curve_style_by_name == {"A": "A", "Z": "Z"}
    assert spec.surface_style_by_name == {"Clay": "Clay", "Sand": "Sand"}
    assert (spec.max_curve_items, spec.max_surface_items) == (18, 18)


@pytest.fixture
def mdpa_svg_input(tmp_path):
    source = tmp_path / "model.mdpa"
    source.write_text(
        """Begin Nodes
1 0 0 0
2 1 0 0
3 0 1 0
4 1 1 0
5 0.5 0 0
End Nodes
Begin Elements Element2D3N
1 0 1 2 3
2 0 2 4 3
End Elements
Begin Conditions LineCondition2D3N
1 0 1 2 5
End Conditions
Begin SubModelPart Soil
Begin SubModelPartElements
1
2
End SubModelPartElements
End SubModelPart
Begin SubModelPart Bottom
Begin SubModelPartConditions
1
End SubModelPartConditions
End SubModelPart
Begin SubModelPart NodesOnly
Begin SubModelPartNodes
1
2
End SubModelPartNodes
End SubModelPart
""",
        encoding="utf-8",
    )
    return source


def test_parse_mdpa_maps_submodelpart_membership_and_quadratic_condition(mdpa_svg_input):
    model = converter.parse_mdpa(mdpa_svg_input)

    assert model.points[5] == (0.5, 0)
    assert model.physical_surfaces == {"Soil": [1, 2]}
    assert set(model.physical_curves) == {"Bottom"}
    assert converter._curve_points_by_name(model, ["Bottom"]) == [
        [(0, 0), (0.5, 0), (1, 0)],
    ]
    assert converter._loop_to_polygon_points(model, 1) == [
        (0, 0), (1, 0), (0, 1), (0, 0),
    ]
    assert "NodesOnly" not in model.physical_curves


def test_mdpa_rendering_preserves_region_borders_legend_and_symbols(
    mdpa_svg_input, tmp_path
):
    output = tmp_path / "mdpa.svg"
    model = converter.parse_mdpa(mdpa_svg_input)

    converter.geo_to_svg(
        model, output, 400, 300, 20,
        legend_cfg={
            "surfaces": [{"name": "Soil", "label": "Ground"}],
            "curves": [{"name": "Bottom", "label": "Fixed base"}],
            "phreatic_line": {"points": [[0, 0.5], [1, 0.5]]},
        },
    )

    root = ET.parse(output).getroot()
    surfaces = root.find(f"{SVG}g[@id='physical_surfaces']")
    assert len(surfaces.findall(f"{SVG}polygon")) == 4
    borders = surfaces.findall(f"{SVG}path")
    assert len(borders) == 1
    assert borders[0].attrib["d"].count("M") == 4
    assert len(root.findall(f"{SVG}g[@id='physical_curves']/{SVG}path")) == 1
    assert root.findall(f"{SVG}g[@id='supports_and_loads']/{SVG}line")
    assert root.find(f"{SVG}g[@id='supports_and_loads']/{SVG}path") is not None
    labels = [node.text for node in root.findall(f"{SVG}g[@id='legend']/{SVG}text")]
    assert "Ground" in labels
    assert "Fixed base" in labels


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

    model = converter.parse_mdpa(source)

    assert model.splines[1] == expected_path
    groups = model.physical_surfaces if is_surface else model.physical_curves
    assert groups == {entity_type: [1]}


def test_parse_mdpa_preserves_multiple_memberships(mdpa_svg_input):
    with mdpa_svg_input.open("a", encoding="utf-8") as stream:
        stream.write(
            "Begin SubModelPart Other\nBegin SubModelPartElements\n1\n"
            "End SubModelPartElements\nEnd SubModelPart\n"
        )

    model = converter.parse_mdpa(mdpa_svg_input)

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

    model = converter.parse_mdpa(source)

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
        converter.parse_mdpa(source)


def test_main_converts_mdpa_with_existing_legend_config(
    mdpa_svg_input, tmp_path, monkeypatch, capsys
):
    source = mdpa_svg_input.with_suffix(".MDPA")
    mdpa_svg_input.rename(source)
    output = tmp_path / "nested" / "mdpa.svg"
    config = tmp_path / "legend.json"
    config.write_text(json.dumps({"legend": {
        "surfaces": [{"name": "Soil", "label": "Ground"}],
    }}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", str(source), "--output", str(output),
        "--width", "400", "--height", "300", "--padding", "20",
        "--legend-config", str(config),
    ])

    converter.main()

    root = ET.parse(output).getroot()
    assert root.attrib["viewBox"] == "0 0 400 300"
    assert any(node.text == "Ground" for node in root.iter())
    stdout = capsys.readouterr().out
    assert f"Wrote SVG: {output}" in stdout
    assert "Physical curves=1, physical surfaces=1" in stdout
