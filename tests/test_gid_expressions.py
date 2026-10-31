import numpy as np
import pytest
from matplotlib.tri import Triangulation

from kratos_visualization_tools.gid.expressions import FieldExpression, calculated_headers
from kratos_visualization_tools.gid.models import ResultHeader


@pytest.fixture
def fields():
    headers = [
        ResultHeader("WATER_PRESSURE", step, "Scalar", "OnNodes", None, 1)
        for step in (1, 2)
    ] + [
        ResultHeader("STRESS", 1, "Matrix", "OnGaussPoints", None, 2),
        ResultHeader("DISPLACEMENT", 1, "Vector", "OnNodes", None, 3),
    ]
    values = {
        "WATER_PRESSURE": ("Scalar", {1: np.array([-1.]), 2: np.array([1.]), 3: np.array([2.])}),
        "STRESS": ("Matrix", {i: np.array([4., 1., 1., 6.]) for i in (1, 2, 3)}),
        "DISPLACEMENT": ("Vector", {i: np.array([3., 4.]) for i in (1, 2, 3)}),
    }
    tri = Triangulation([0, 1, 0], [0, 0, 1], [[0, 1, 2]])
    return headers, values, tri


@pytest.mark.parametrize("formula, expected", [
    ("water_pressure", [-1, 1, 2]),
    ("max(abs(WATER_PRESSURE), 1.5)", [1.5, 1.5, 2]),
    ("min(WATER_PRESSURE * 2 + 1, 3)", [-1, 3, 3]),
    ("(stress.xx + stress.yy) / 2", [5, 5, 5]),
    ("stress.trace - WATER_PRESSURE", [11, 9, 8]),
    ("abs(DISPLACEMENT)", [5, 5, 5]),
    ("mag(DISPLACEMENT)", [5, 5, 5]),
    ("DISPLACEMENT.x + DISPLACEMENT.y", [7, 7, 7]),
    ("grad_x(WATER_PRESSURE)", [2, 2, 2]),
    ("grad_y(WATER_PRESSURE)", [3, 3, 3]),
    ("grad(WATER_PRESSURE)", [np.sqrt(13)] * 3),
    ("-WATER_PRESSURE + +1", [2, 0, -1]),
])
def test_formula_numeric_results(fields, formula, expected):
    headers, values, tri = fields
    expression = FieldExpression(formula, headers)
    result = expression.evaluate([1, 2, 3], lambda name, loc: values[name], tri)
    np.testing.assert_allclose([result[i][0] for i in (1, 2, 3)], expected)


@pytest.mark.parametrize("formula", [
    "__import__('os')", "WATER_PRESSURE.__class__", "WATER_PRESSURE[0]",
    "WATER_PRESSURE ** 2", "open('file')", "lambda: WATER_PRESSURE",
    "min(WATER_PRESSURE)", "max(WATER_PRESSURE, x=1)", "True + WATER_PRESSURE",
    "unknown + 1", "WATER_PRESSURE +", "42", "1e400 + WATER_PRESSURE",
])
def test_invalid_formula_rejected(fields, formula):
    with pytest.raises(ValueError):
        FieldExpression(formula, fields[0])


@pytest.mark.parametrize("formula, message", [
    ("WATER_PRESSURE / 0", "Division by zero"),
    ("STRESS", "tensor component"),
    ("DISPLACEMENT + 1", "vector component"),
    ("WATER_PRESSURE.xx", "not available"),
    ("STRESS.x", "not available"),
])
def test_runtime_errors_explicit(fields, formula, message):
    headers, values, tri = fields
    with pytest.raises(ValueError, match=message):
        FieldExpression(formula, headers).evaluate([1, 2, 3], lambda name, loc: values[name], tri)


def test_same_step_selection_and_projected_location(fields):
    expression = FieldExpression("STRESS.xx + WATER_PRESSURE", fields[0])
    assert expression.steps == [1]
    assert expression.locations == {"STRESS": "OnGaussPoints", "WATER_PRESSURE": "OnNodes"}
    headers = calculated_headers(fields[0], {"TOTAL": expression.text})
    assert [(h.name, h.step, h.location) for h in headers if h.header_line < 0] == [
        ("TOTAL", 1, "OnNodes")]
    # Definitions stay stored, but are not selectable when a stage lacks inputs.
    assert calculated_headers(fields[0][:2], {"TOTAL": expression.text}) == fields[0][:2]


def test_sparse_ids_are_aligned_not_zipped(fields):
    headers, values, tri = fields
    del values["STRESS"][1][2]
    result = FieldExpression("STRESS.xx + WATER_PRESSURE", headers).evaluate(
        [1, 2, 3], lambda name, loc: values[name], tri)
    assert set(result) == {1, 3}
    np.testing.assert_allclose([result[1][0], result[3][0]], [3, 6])


def test_gradient_requires_triangulation(fields):
    with pytest.raises(ValueError, match="triangulation"):
        FieldExpression("grad(WATER_PRESSURE)", fields[0]).evaluate(
            [1, 2, 3], lambda name, loc: fields[1][name], None)


def test_no_shared_steps_disables_calculated_header(fields):
    fields[0][2].step = 3
    expression = FieldExpression("STRESS.xx + WATER_PRESSURE", fields[0])
    assert expression.steps == []
    assert calculated_headers(fields[0], {"TOTAL": expression.text}) == fields[0]


def test_invalid_settings_formula_is_not_silently_ignored(fields):
    with pytest.raises(ValueError):
        calculated_headers(fields[0], {"BAD": "__import__('os') + unknown"})
