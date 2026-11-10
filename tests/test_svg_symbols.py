import xml.etree.ElementTree as ET

import pytest

from kratos_visualization_tools.svg import document, symbols


SVG = f"{{{document.SVG_NS}}}"


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

    samples = symbols._collect_vertical_side_samples(polylines, side_sign)

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

    assert symbols._pick_two_samples(candidates) == [candidates[i] for i in indices]
    assert candidates == original


@pytest.mark.parametrize("polylines", [[], [[]], [[(0, 0)]], [[(0, 0), (0, 0)]]])
@pytest.mark.parametrize("renderer", ["hatching", "fixity"])
def test_support_drawers_skip_empty_or_degenerate_geometry(polylines, renderer):
    group = ET.Element(f"{SVG}g")

    if renderer == "hatching":
        symbols._draw_bottom_hatching(group, polylines, center_svg=(0, 10))
    else:
        symbols._draw_side_fixity(group, polylines, outward_sign=-1)

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

    symbols._draw_bottom_hatching(
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

    symbols._draw_side_fixity(
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

    symbols._draw_side_fixity(
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

    symbols._draw_load_arrows(group, polylines)

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

    symbols._draw_load_arrows(group, [poly], spacing=10, arrow_len=8)

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

    symbols._draw_load_arrows(
        group, [[(0, 20), (10, 20)], [(30, 40), (40, 40)]]
    )

    assert len(group.findall(f"{SVG}polygon")) == 4
    lines = group.findall(f"{SVG}line")
    assert len(lines) == 6
    assert (lines[2].attrib["x1"], lines[2].attrib["x2"]) == ("0.000", "10.000")
    assert (lines[5].attrib["x1"], lines[5].attrib["x2"]) == ("30.000", "40.000")


@pytest.mark.parametrize("points", [[], [(0, 0)]])
def test_draw_phreatic_line_skips_incomplete_geometry(points):
    group = ET.Element(f"{SVG}g")

    symbols._draw_phreatic_line(group, points)

    assert len(group) == 0


@pytest.mark.parametrize(
    ("span", "target_count", "expected"),
    [(10, 5, 2), (100, 3, 50), (1, 4, 0.5), (7, 1, 10), (0, 3, 5e-13)],
)
def test_nice_tick_step_rounds_to_one_two_five_steps(span, target_count, expected):
    assert symbols._nice_tick_step(span, target_count) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value", "step", "expected"),
    [(10, 5, "10"), (0.5, 0.5, "0.5"), (2.5, 0.25, "2.50"), (-1e-17, 0.25, "0.00")],
)
def test_format_tick_value_uses_step_precision(value, step, expected):
    assert symbols._format_tick_value(value, step) == expected
