import pytest

from kratos_visualization_tools.svg import geometry, models


def test_loop_polygon_follows_curve_direction_and_closes():
    model = models.GeoModel(
        points={1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)},
        lines={1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)},
        curve_loops={1: [1, 2, 3, 4]},
    )

    assert geometry._loop_to_polygon_points(model, 1) == [
        (0, 0), (1, 0), (1, 1), (0, 1), (0, 0),
    ]
    assert geometry._entity_polyline(model, -1) == [(1, 0), (0, 0)]


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
    assert geometry._entity_polyline(curve_model, signed_id) == expected


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

    assert geometry._loop_to_polygon_points(curve_model, 1) == expected


def test_loop_polygon_returns_empty_for_unknown_loop(curve_model):
    assert geometry._loop_to_polygon_points(curve_model, 99) == []


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
    assert geometry._curve_points_by_name(curve_model, names) == expected


def test_curve_points_by_name_chains_unordered_and_duplicate_edges():
    model = models.GeoModel(
        points={i: (float(i), 0.0) for i in range(5)},
        splines={1: [2, 3], 2: [0, 1], 3: [4, 3], 4: [1, 2], 5: [1, 0]},
        physical_curves={"Bottom": [1, 2, 3, 4, 5]},
    )

    assert geometry._curve_points_by_name(model, ["Bottom"]) == [
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
    ]


@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [(0, 0, 0), (3, 4, 5), (-3, -4, 5), (0, -2, 2), (1e200, 0, 1e200)],
)
def test_line_len_returns_euclidean_length(dx, dy, expected):
    assert geometry._line_len(dx, dy) == pytest.approx(expected)


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
    samples = geometry._sample_polyline_points(poly, spacing)

    assert len(samples) == len(expected)
    for sample, expected_sample in zip(samples, expected):
        assert sample == pytest.approx(expected_sample)


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
    transform = geometry._svg_transform(*bounds, width=300, height=200, padding=10)

    assert transform(point) == pytest.approx(expected)


def test_entity_polyline_prefers_line_when_spline_has_same_id(curve_model):
    curve_model.splines[1] = [3, 4]

    assert geometry._entity_polyline(curve_model, 1) == [(0, 0), (2, 0)]


def test_chain_segments_joins_open_and_closed_chains():
    segments = [
        ((1, 0), (2, 0)), ((0, 0), (1, 0)),
        ((5, 5), (6, 5)), ((6, 5), (6, 6)), ((6, 6), (5, 5)),
    ]

    chains = geometry._chain_segments(segments)

    assert sorted(len(chain) for chain in chains) == [3, 4]
    open_chain = next(chain for chain in chains if len(chain) == 3)
    assert {open_chain[0], open_chain[-1]} == {(0, 0), (2, 0)}
    closed_chain = next(chain for chain in chains if len(chain) == 4)
    assert closed_chain[0] == closed_chain[-1]


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(0, 0), (1, 0)], [(0, 0), (1, 0)]),
        ([(0, 0), (1, 0.01), (2, 0)], [(0, 0), (2, 0)]),
        ([(0, 0), (1, 1), (2, 0)], [(0, 0), (1, 1), (2, 0)]),
        ([(0, 0), (2, 0), (1, 0)], [(0, 0), (2, 0), (1, 0)]),
        ([(0, 0), (0, 0), (1, 0)], [(0, 0), (1, 0)]),
        (
            [(1, 0), (2, 0), (2, 1), (0, 1), (0, 0), (1, 0)],
            [(2, 0), (2, 1), (0, 1), (0, 0), (2, 0)],
        ),
    ],
)
def test_drop_collinear_removes_only_points_on_straight_runs(points, expected):
    assert geometry._drop_collinear(points) == expected


def test_path_data_closes_loops_and_skips_degenerate_polylines():
    assert geometry._path_data([
        [(0, 0), (1, 0), (1, 1), (0, 0)],
        [(5, 5), (5.01, 5.01)],
        [(0, 2), (1, 2)],
    ]) == "M0.0,0.0L1.0,0.0L1.0,1.0ZM0.0,2.0L1.0,2.0"
