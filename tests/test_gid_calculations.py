import numpy as np
import pytest

import visualize_gid_results as viz


def test_cycle_sequence_wraps_and_falls_back():
    assert viz.cycle_sequence(("x", "y", "z"), "z") == "x"
    assert viz.cycle_sequence(("x", "y", "z"), "y", -1) == "x"
    assert viz.cycle_sequence(("x", "y", "z"), "missing") == "x"
    assert viz.cycle_sequence(("x", "y"), "x", 1) == "y"


def test_validate_component_accepts_supported_and_rejects_invalid():
    assert viz.validate_component("y", ("x", "y", "z"), "component") == "y"
    with pytest.raises(ValueError, match="Unsupported component 'w'"):
        viz.validate_component("w", ("x", "y"), "component")


def test_shared_scale_uses_values_from_both_solutions():
    assert viz.calculate_shared_scale(
        [np.array([1.0, np.nan]), np.array([-2.0, 4.0])]
    ) == (-2.0, 4.0)
    assert viz.calculate_shared_scale(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])], vmin=0.0, vmax=5.0
    ) == (0.0, 5.0)

    with pytest.raises(ValueError, match="minimum"):
        viz.calculate_shared_scale([np.array([1.0])], vmin=2.0, vmax=1.0)


def test_difference_interpolation_and_symmetric_scale():
    source_field = {
        "points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "scalar": np.array([10.0, 20.0, 30.0]),
    }
    interpolated = viz.interpolate_scalar_field(
        source_field, np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
    )

    np.testing.assert_allclose(interpolated, [20.0, 10.0])
    assert viz.calculate_symmetric_scale(np.array([-2.0, 5.0])) == (-5.0, 5.0)


def test_project_gauss_values_to_surface_and_line_nodes():
    projected = viz.project_element_to_nodes(
        {
            10: np.array([2.0, 4.0]),
            11: np.array([6.0, 8.0]),
            20: np.array([4.0, 6.0]),
            21: np.array([1.0, 2.0, 3.0]),
            22: np.array([8.0, 10.0]),
            23: np.array([10.0, 12.0]),
        },
        [
            (10, [1, 2, 3]),
            (11, [1, 2, 3]),
            (30, [4, 5]),
        ],
        [
            (30, [1], 1),
            (31, [1], 1),
            (20, [1, 3], 1),
            (21, [2, 3], 1),
            (22, [4, 5], 1),
            (23, [4, 5], 1),
        ],
    )

    np.testing.assert_allclose(projected[1], [4.0, 6.0])
    np.testing.assert_allclose(projected[2], [4.0, 6.0])
    np.testing.assert_allclose(projected[3], [4.0, 6.0])


def test_gauss_point_blocks_are_grouped_and_extrapolated_to_linear_element_nodes(tmp_path):
    result_path = tmp_path / "gauss_point_fields.post.res"
    quad_offset = 1.0 / np.sqrt(3.0)
    result_path.write_text(
        'GaussPoints "tri3_gp" ElemType Triangle\n'
        "Number Of Gauss Points: 3\n"
        "Natural Coordinates: Given\n"
        "0.166667 0.166667\n"
        "0.666667 0.166667\n"
        "0.166667 0.666667\n"
        "End GaussPoints\n"
        'GaussPoints "quad4_gp" ElemType Quadrilateral\n'
        "Number Of Gauss Points: 4\n"
        "Natural Coordinates: Given\n"
        f"{-quad_offset} {-quad_offset}\n"
        f"{quad_offset} {-quad_offset}\n"
        f"{quad_offset} {quad_offset}\n"
        f"{-quad_offset} {quad_offset}\n"
        "End GaussPoints\n"
        'Result "LINEAR_FIELD" "Load" 1 Scalar OnGaussPoints "tri3_gp"\n'
        "Values\n"
        "10 0.5\n"
        "1.0\n"
        "1.5\n"
        "End Values\n"
        'Result "LINEAR_FIELD" "Load" 1 Scalar OnGaussPoints "quad4_gp"\n'
        "Values\n"
        f"20 {-quad_offset}\n"
        f"{quad_offset}\n"
        f"{quad_offset}\n"
        f"{-quad_offset}\n"
        "End Values\n",
        encoding="utf-8",
    )

    result_type, gauss_values = viz.load_values(
        result_path, "LINEAR_FIELD", 1, "OnGaussPoints"
    )
    assert result_type == "Scalar"
    np.testing.assert_allclose(gauss_values[10][:, 0], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(
        gauss_values[20][:, 0], [-quad_offset, quad_offset, quad_offset, -quad_offset]
    )

    projected = viz.project_element_to_nodes(
        gauss_values, [(10, [1, 2, 3]), (20, [4, 5, 6, 7])], None
    )
    np.testing.assert_allclose(
        [projected[node_id][0] for node_id in (1, 2, 3)], [0, 1, 2], atol=1e-14
    )
    np.testing.assert_allclose(
        [projected[node_id][0] for node_id in (4, 5, 6, 7)],
        [-1, 1, 1, -1],
        atol=1e-14,
    )


def test_components_and_gradient_handle_supported_data():
    assert viz.vector_component(np.array([3.0, 4.0]), "magnitude") == 5.0
    assert viz.tensor_component(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "trace") == 6.0
    assert viz.field_scalar_value(np.array([1.0, 2.0]), "Vector", "y", "trace") == 2.0

    gradient = viz.finite_difference_gradient([2.0, 0.0, 1.0], [4.0, 0.0, 2.0])
    np.testing.assert_allclose(gradient, [2.0, 2.0, 2.0])

    with pytest.raises(ValueError, match="Unsupported component"):
        viz.vector_component(np.array([1.0]), "bad")


def test_tensor_component_maps_six_component_tensor():
    tensor = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert viz.tensor_component(tensor, "xx") == 1.0
    assert viz.tensor_component(tensor, "yy") == 2.0
    assert viz.tensor_component(tensor, "zz") == 3.0
    assert viz.tensor_component(tensor, "xy") == 4.0
    assert viz.tensor_component(tensor, "yx") == 4.0
    assert viz.tensor_component(tensor, "yz") == 5.0
    assert viz.tensor_component(tensor, "zy") == 5.0
    assert viz.tensor_component(tensor, "xz") == 6.0
    assert viz.tensor_component(tensor, "zx") == 6.0
    assert viz.tensor_component(tensor, "trace") == 6.0
    assert viz.tensor_component(tensor, "magnitude") == pytest.approx(np.linalg.norm(tensor))


def test_tensor_component_maps_flat_square_matrix_and_handles_invalid_input():
    matrix = np.arange(1.0, 10.0)

    assert viz.tensor_component(matrix, "xx") == 1.0
    assert viz.tensor_component(matrix, "xy") == 2.0
    assert viz.tensor_component(matrix, "yx") == 4.0
    assert viz.tensor_component(matrix, "yy") == 5.0
    assert viz.tensor_component(matrix, "zz") == 9.0
    assert viz.tensor_component(matrix, "trace") == 15.0

    assert np.isnan(viz.tensor_component(np.array([]), "xx"))
    with pytest.raises(ValueError, match="Unsupported tensor component"):
        viz.tensor_component(matrix, "bad")


def test_tensor_component_handles_short_flattened_arrays():
    three_values = np.array([1.0, 2.0, 3.0])
    assert viz.tensor_component(three_values, "xx") == 1.0
    assert viz.tensor_component(three_values, "xy") == 2.0
    assert viz.tensor_component(three_values, "yx") == 3.0
    assert np.isnan(viz.tensor_component(three_values, "yy"))
    assert viz.tensor_component(three_values, "zz") == 3.0
    assert viz.tensor_component(three_values, "trace") == 6.0

    two_values = np.array([4.0, 5.0])
    assert viz.tensor_component(two_values, "xx") == 4.0
    assert viz.tensor_component(two_values, "xy") == 5.0
    assert np.isnan(viz.tensor_component(two_values, "yx"))
    assert np.isnan(viz.tensor_component(two_values, "yy"))
    assert np.isnan(viz.tensor_component(two_values, "zz"))
    assert np.isnan(viz.tensor_component(two_values, "trace"))

    two_by_two = np.array([1.0, 2.0, 3.0, 4.0])
    assert viz.tensor_component(two_by_two, "zz") == 3.0
    assert viz.tensor_component(two_by_two, "trace") == 5.0


def test_compute_nodal_gradient_recovers_linear_field_gradient():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = viz.mtri.Triangulation(points[:, 0], points[:, 1], np.array([[0, 1, 2]]))
    nodal_scalar = 5.0 + 2.0 * points[:, 0] + 3.0 * points[:, 1]

    grad_x, grad_y = viz.compute_nodal_gradient(tri, nodal_scalar)

    np.testing.assert_allclose(grad_x, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(grad_y, [3.0, 3.0, 3.0])


def test_compute_nodal_gradient_skips_invalid_and_degenerate_triangles():
    points = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
    ])
    tri = viz.mtri.Triangulation(
        points[:, 0], points[:, 1], np.array([[0, 1, 2], [3, 4, 5]])
    )

    invalid_values = np.array([1.0, np.nan, 3.0, 1.0, 2.0, 3.0])
    grad_x, grad_y = viz.compute_nodal_gradient(tri, invalid_values)
    assert np.all(np.isnan(grad_x))
    assert np.all(np.isnan(grad_y))

    valid_values = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    grad_x, grad_y = viz.compute_nodal_gradient(tri, valid_values)
    np.testing.assert_allclose(grad_x[:3], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(grad_y[:3], [2.0, 2.0, 2.0])
    assert np.all(np.isnan(grad_x[3:]))
    assert np.all(np.isnan(grad_y[3:]))


def test_compute_nodal_gradient_returns_nan_without_triangles():
    nodal_scalar = np.array([1.0, 2.0, 3.0])

    grad_x, grad_y = viz.compute_nodal_gradient(None, nodal_scalar)

    assert np.all(np.isnan(grad_x))
    assert np.all(np.isnan(grad_y))


def test_set_profile_data_stores_arrays_and_computes_gradients():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)

    viewer._set_profile_data([0, 1, 2], [1, 3, 5], profile_x_current=0.25)

    np.testing.assert_array_equal(viewer.profile_y_data, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(viewer.profile_values, [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(viewer.profile_gradients, [2.0, 2.0, 2.0])
    assert viewer.profile_x_current == 0.25


def test_parse_line_spec_accepts_line_points_and_line_samples():
    assert viz._parse_line_spec({
        "line_points": [[1, 2], [3, 4]],
        "line_samples": 5,
    }) == ((1.0, 2.0), (3.0, 4.0), 5)

    with pytest.raises(ValueError, match="line coordinates"):
        viz._parse_line_spec({})


def test_sample_line_values_idw_handles_exact_and_weighted_samples():
    nodes = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
    }
    values = {1: 10.0, 2: 20.0, 3: 30.0}

    samples = viz._sample_line_values_idw(nodes, values, (0.0, 0.0), (0.5, 0.5), 2)

    assert samples[0][-1] == 10.0
    assert samples[1][-1] == pytest.approx(20.0)


def test_sample_vertical_profile_interpolates_values_inside_triangle():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = viz.mtri.Triangulation(points[:, 0], points[:, 1], np.array([[0, 1, 2]]))
    nodal_scalar = points[:, 0] + 2.0 * points[:, 1]

    ys, values = viz.sample_vertical_profile_from_triangulation(
        tri, nodal_scalar, profile_x=0.25, y_min=0.0, y_max=0.75, n_samples=4
    )

    np.testing.assert_allclose(ys, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(values, 0.25 + 2.0 * ys)


def test_sample_vertical_profile_returns_nan_outside_domain_and_for_invalid_data():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = viz.mtri.Triangulation(points[:, 0], points[:, 1], np.array([[0, 1, 2]]))

    _, outside_values = viz.sample_vertical_profile_from_triangulation(
        tri, np.array([0.0, 1.0, 2.0]), profile_x=2.0, y_min=0.0, y_max=1.0, n_samples=3
    )
    assert np.all(np.isnan(outside_values))

    _, invalid_values = viz.sample_vertical_profile_from_triangulation(
        tri, np.array([0.0, np.nan, 2.0]), profile_x=0.25, y_min=0.0, y_max=0.5, n_samples=3
    )
    assert np.all(np.isnan(invalid_values))

    non_square = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert viz.tensor_component(non_square, "trace") == 5.0


def test_parse_profile_path_accepts_segments_and_polylines():
    assert viz.parse_profile_path("0,0;1,0") == [(0.0, 0.0), (1.0, 0.0)]
    assert viz.parse_profile_path([[0, 0], [1, 0], [1, 1]]) == [
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)
    ]
    with pytest.raises(ValueError, match="at least two"):
        viz.parse_profile_path("0,0")
    with pytest.raises(ValueError, match="formatted as x,y"):
        viz.parse_profile_path("0,0;invalid")


def test_sample_profile_path_interpolates_segment_and_polyline_values():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tri = viz.mtri.Triangulation(points[:, 0], points[:, 1], np.array([[0, 1, 2]]))
    nodal_scalar = points[:, 0] + 2.0 * points[:, 1]

    distance, values, sampled_points = viz.sample_profile_path_from_triangulation(
        tri, nodal_scalar, [(0.0, 0.0), (0.5, 0.0)], n_samples=3
    )

    np.testing.assert_allclose(distance, [0.0, 0.25, 0.5])
    np.testing.assert_allclose(sampled_points[:, 1], 0.0)
    np.testing.assert_allclose(values, distance)

    polyline_distance, _, polyline_points = viz.sample_profile_path_from_triangulation(
        tri, nodal_scalar, [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)], n_samples=5
    )
    assert polyline_distance[-1] == pytest.approx(0.5 + np.sqrt(0.5))
    np.testing.assert_allclose(polyline_points[0], [0.0, 0.0])
    np.testing.assert_allclose(polyline_points[-1], [0.0, 0.5], atol=1e-12)
