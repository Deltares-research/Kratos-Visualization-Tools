import pytest

from kratos_visualization_tools.svg import geo_reader


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
    assert geo_reader.eval_expr(expression, names) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["missing + 1", "__import__('os')", "a[0]", "a // 2", "'text'"],
)
def test_eval_expr_rejects_unknown_or_unsupported_expressions(expression):
    with pytest.raises((ValueError, SyntaxError)):
        geo_reader.eval_expr(expression, {"a": 2})


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

    model = geo_reader.parse_geo(path)

    assert model.vars["side"] == 2
    assert model.points[2] == (2, 0)
    assert model.lines[1] == (1, 2)
    assert model.splines[3] == [3, 4]
    assert model.curve_loops[1] == [1, 2, 3, 4]
    assert model.plane_surfaces[1] == [1]
    assert model.physical_curves["Boundary"] == [1, -2]
    assert model.physical_surfaces["Soil"] == [1]


def test_eval_expr_rejects_unsupported_unary_operator():
    with pytest.raises(ValueError, match="Unsupported unary operator"):
        geo_reader.eval_expr("~2", {})


def test_parse_geo_handles_unterminated_statements_and_invalid_definitions(tmp_path):
    path = tmp_path / "partial.geo"
    path.write_text(
        "invalid = unknown;\nPoint(9) = {1};\n"
        "Point(1) = {Pi, 2};\nPoint(2) = {0, 0}",
        encoding="utf-8",
    )

    model = geo_reader.parse_geo(path)

    assert "invalid" not in model.vars
    assert model.points == {1: (model.vars["Pi"], 2), 2: (0, 0)}
