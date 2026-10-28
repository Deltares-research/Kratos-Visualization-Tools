import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import visualize_gid_results as viz


def test_legacy_export_helper_patch_reaches_package_calls(monkeypatch, tmp_path):
    from kratos_visualization_tools import gid

    expected = tmp_path / "patched.csv"
    original = viz.build_output_path
    with monkeypatch.context() as patch:
        patch.setattr(viz, "build_output_path", lambda *args: expected)
        assert gid.build_result_export_path("pressure", "1", "OnNodes", "field") == expected
        assert viz.build_result_export_path("pressure", "1", "OnNodes", "field") == expected
    assert gid.build_output_path is original


def test_legacy_class_patch_reaches_all_rendering_imports(monkeypatch):
    from kratos_visualization_tools.gid import cli, comparison, plotting

    original = viz.ComparisonViewer
    replacement = object()
    with monkeypatch.context() as patch:
        patch.setattr(viz, "ComparisonViewer", replacement)
        assert cli.ComparisonViewer is replacement
        assert plotting.ComparisonViewer is replacement
        assert comparison.ComparisonViewer is replacement
    assert cli.ComparisonViewer is original
    assert plotting.ComparisonViewer is original


def test_cycle_sequence_wraps_and_falls_back():
    assert viz.cycle_sequence(("x", "y", "z"), "z") == "x"
    assert viz.cycle_sequence(("x", "y", "z"), "y", -1) == "x"
    assert viz.cycle_sequence(("x", "y", "z"), "missing") == "x"
    assert viz.cycle_sequence(("x", "y"), "x", 1) == "y"


def test_validate_component_accepts_supported_and_rejects_invalid():
    assert viz.validate_component("y", ("x", "y", "z"), "component") == "y"
    with pytest.raises(ValueError, match="Unsupported component 'w'"):
        viz.validate_component("w", ("x", "y"), "component")


def write_mesh(path):
    path.write_text(
        'MESH "Surface" dimension 2 ElemType Triangle Nnode 3\n'
        "Coordinates\n"
        "1 0.0 0.0 0.0\n"
        "2 1.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
        "End Coordinates\n"
        "Elements\n"
        "10 1 2 3 1\n"
        "End Elements\n"
        'MESH "Profile" dimension 2 ElemType Linear Nnode 2\n'
        "Coordinates\n"
        "1 0.0 0.0 0.0\n"
        "3 0.0 1.0 0.0\n"
        "End Coordinates\n"
        "Elements\n"
        "20 1 3 1\n"
        "End Elements\n",
        encoding="utf-8",
    )


def write_results(path):
    path.write_text(
        'Result "PRESSURE" "Load" 1 Scalar OnNodes\n'
        "Values\n"
        "1 0.0\n"
        "2 1.0\n"
        "3 2.0\n"
        "End Values\n"
        'Result "PRESSURE" "Load" 2 Scalar OnNodes\n'
        "Values\n"
        "1 10.0\n"
        "2 11.0\n"
        "3 12.0\n"
        "End Values\n"
        'Result "DISPLACEMENT" "Load" 2 Vector OnNodes\n'
        "Values\n"
        "1 3.0 4.0\n"
        "2 0.0 2.0\n"
        "End Values\n"
        'Result "STRESS" "Load" 2 Matrix OnGaussPoints "GP"\n'
        "Values\n"
        "10 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "End Values\n",
        encoding="utf-8",
    )


def write_second_mesh(path):
    path.write_text(
        'MESH "OtherSurface" dimension 2 ElemType Triangle Nnode 3\n'
        "Coordinates\n"
        "101 0.0 0.0 0.0\n"
        "102 2.0 0.0 0.0\n"
        "103 0.0 2.0 0.0\n"
        "End Coordinates\n"
        "Elements\n"
        "110 101 102 103 1\n"
        "End Elements\n",
        encoding="utf-8",
    )


def write_second_results(path):
    path.write_text(
        'Result "PRESSURE" "OtherLoad" 1 Scalar OnNodes\n'
        "Values\n"
        "101 20.0\n"
        "102 21.0\n"
        "103 22.0\n"
        "End Values\n"
        'Result "DISPLACEMENT" "OtherLoad" 1 Vector OnNodes\n'
        "Values\n"
        "101 1.0 0.0\n"
        "102 2.0 0.0\n"
        "103 3.0 0.0\n"
        "End Values\n"
        'Result "TEMPERATURE" "OtherLoad" 1 Scalar OnNodes\n'
        "Values\n"
        "101 100.0\n"
        "End Values\n",
        encoding="utf-8",
    )


def write_active_mdpa(path, submodel_name="active", include_entities=True):
    content = [
        f"Begin SubModelPart {submodel_name}\n",
        "Begin SubModelPartNodes\n",
        "1\n",
        "3\n",
        "End SubModelPartNodes\n",
    ]
    if include_entities:
        content.extend([
            "Begin SubModelPartElements\n",
            "10\n",
            "End SubModelPartElements\n",
        ])
    content.append("End SubModelPart\n")
    path.write_text("".join(content), encoding="utf-8")


def test_parse_mesh_collects_blocks_and_plot_connectivity(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    write_mesh(mesh_path)

    all_nodes, triangles, surface_elements, line_meshes, block_nodes = viz.parse_post_msh(mesh_path)

    assert all_nodes[2] == (1.0, 0.0, 0.0)
    assert triangles == [(1, 2, 3)]
    assert surface_elements == [(10, [1, 2, 3])]
    assert line_meshes == [("Profile", {1, 3}, [(20, [1, 3], 1)])]
    assert block_nodes == {"Surface": {1, 2, 3}, "Profile": {1, 3}}


def test_parse_mesh_recovers_nodes_without_coordinates_and_splits_quadrilateral(tmp_path):
    mesh_path = tmp_path / "quadrilateral.post.msh"
    mesh_path.write_text(
        'MESH "Quad" dimension 2 ElemType Quadrilateral Nnode 4\n'
        "Elements\n"
        "20 1 2 3 4 1\n"
        "End Elements\n",
        encoding="utf-8",
    )

    all_nodes, triangles, surface_elements, line_meshes, block_nodes = viz.parse_post_msh(mesh_path)

    assert all_nodes == {}
    assert triangles == [(1, 2, 3), (1, 3, 4)]
    assert surface_elements == [(20, [1, 2, 3, 4])]
    assert line_meshes == []
    assert block_nodes == {"Quad": {1, 2, 3, 4}}


def test_interactive_viewer_initializes_single_solution_options(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)

    viewer = viz.InteractiveViewer(
        headers=headers,
        res_path=result_path,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements + [(99, [1, 2, 3, 1])],
        line_meshes=line_meshes,
        profile_nodes={1, 3},
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
        show_isolines=True,
        show_profile=False,
        contour_levels=7,
    )
    try:
        assert viewer.current_result == "PRESSURE"
        assert viewer.show_isolines
        assert not viewer.show_profile
        assert viewer.contour_levels == 7
        assert viewer.ax1 is None
        assert viewer.gauss_points_by_id[99] == (0.25, 0.25)
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_update_plot_refreshes_mesh(tmp_path):
    viewer = make_single_viewer(tmp_path, initial_location=None)
    try:
        viewer.update_plot()
        assert viewer.ax0.get_title() == "PRESSURE @ step 2"

        viewer.show_mesh = True
        viewer.update_plot()
        assert viewer.ax0.lines
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_keeps_plot_area_when_result_scale_changes(tmp_path):
    viewer = make_single_viewer(tmp_path, initial_location=None)
    try:
        initial_bounds = viewer.ax0.get_position().bounds
        result_name = "WIDE_RANGE"
        viewer.headers.append(viz.ResultHeader(result_name, 1, "Scalar", "OnNodes", None, 0))
        viewer.result_names.append(result_name)
        viewer.result_cache[(result_name, 1, "OnNodes")] = (
            "Scalar",
            {
                1: np.array([1.0e-12]),
                2: np.array([1.0e8]),
                3: np.array([1.0e15]),
            },
        )

        viewer.current_result = result_name
        viewer.current_step_str = "latest"
        viewer.current_location = None
        viewer.update_plot()

        np.testing.assert_allclose(viewer.ax0.get_position().bounds, initial_bounds)
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_update_plot_refreshes_gradient(tmp_path):
    viewer = make_single_viewer(tmp_path, initial_location=None)
    try:
        viewer.show_gradient = True
        viewer.update_plot()
        assert viewer.current_field_grad_x is not None
        assert viewer.current_field_grad_y is not None
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_update_plot_shows_missing_result_error(tmp_path):
    viewer = make_single_viewer(tmp_path, initial_location=None)
    try:
        viewer.current_result = "MISSING_RESULT"
        viewer.update_plot()
        assert any("Error:" in text.get_text() for text in viewer.ax0.texts)
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_gradient_without_triangulation_shows_error(tmp_path):
    viewer = make_single_viewer(tmp_path, triangles=[])
    try:
        viewer.show_gradient = True
        viewer.update_plot()
        assert viewer.ax0.axison is False
        assert any("Gradient visualization requires" in text.get_text() for text in viewer.ax0.texts)
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_matrix_gradient_falls_back_to_tensor_magnitude(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "matrix.post.res"
    write_mesh(mesh_path)
    result_path.write_text(
        'Result "STRESS" "Load" 1 Matrix OnNodes\n'
        "Values\n"
        "1 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "2 2.0 3.0 4.0 5.0 6.0 7.0\n"
        "3 3.0 4.0 5.0 6.0 7.0 8.0\n"
        "End Values\n",
        encoding="utf-8",
    )
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)
    viewer = viz.InteractiveViewer(
        headers=headers,
        res_path=result_path,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements,
        line_meshes=line_meshes,
        profile_nodes=set(),
        initial_result="STRESS",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=viz.ActiveSubmodelPart("empty", set(), set()),
    )
    try:
        viewer.show_gradient = True
        viewer.update_plot()
        assert viewer.current_field_source_values is not None
        assert viewer.current_field_grad_magnitude is not None
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_update_plot_handles_node_vector_and_matrix_fields(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "node_fields.post.res"
    write_mesh(mesh_path)
    result_path.write_text(
        'Result "DISPLACEMENT" "Load" 1 Vector OnNodes\n'
        "Values\n"
        "1 1.0 2.0\n"
        "2 3.0 4.0\n"
        "3 5.0 6.0\n"
        "End Values\n"
        'Result "STRESS" "Load" 1 Matrix OnNodes\n'
        "Values\n"
        "1 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "2 2.0 3.0 4.0 5.0 6.0 7.0\n"
        "3 3.0 4.0 5.0 6.0 7.0 8.0\n"
        "End Values\n",
        encoding="utf-8",
    )
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)
    viewers = []
    try:
        for result_name in ("DISPLACEMENT", "STRESS"):
            viewer = viz.InteractiveViewer(
                headers=headers,
                res_path=result_path,
                all_nodes=all_nodes,
                triangles=triangles,
                surface_elements=surface_elements,
                line_meshes=line_meshes,
                profile_nodes=set(),
                initial_result=result_name,
                initial_step="latest",
                initial_location="OnNodes",
                profile_x=0.0,
                x_tol=0.35,
                initial_tensor_component="trace",
                active_filter=None,
            )
            assert viewer.current_location == "OnNodes"
            viewers.append(viewer)
    finally:
        for viewer in viewers:
            close_viewer_figures(viewer)


def test_interactive_viewer_update_plot_uses_discrete_profile_fallback(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "profile_fields.post.res"
    write_mesh(mesh_path)
    result_path.write_text(
        'Result "PRESSURE" "Load" 1 Scalar OnNodes\n'
        "Values\n"
        "1 10.0\n"
        "2 11.0\n"
        "3 12.0\n"
        "End Values\n"
        'Result "DISPLACEMENT" "Load" 1 Vector OnNodes\n'
        "Values\n"
        "1 1.0 2.0\n"
        "2 3.0 4.0\n"
        "3 5.0 6.0\n"
        "End Values\n"
        'Result "STRESS" "Load" 1 Matrix OnNodes\n'
        "Values\n"
        "1 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "2 2.0 3.0 4.0 5.0 6.0 7.0\n"
        "3 3.0 4.0 5.0 6.0 7.0 8.0\n"
        "End Values\n",
        encoding="utf-8",
    )
    all_nodes, _, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)
    viewers = []
    try:
        for result_name in ("PRESSURE", "DISPLACEMENT", "STRESS"):
            viewer = viz.InteractiveViewer(
                headers=headers,
                res_path=result_path,
                all_nodes=all_nodes,
                triangles=[],
                surface_elements=surface_elements,
                line_meshes=line_meshes,
                profile_nodes={1, 3},
                initial_result=result_name,
                initial_step="latest",
                initial_location="OnNodes",
                profile_x=0.0,
                x_tol=0.35,
                initial_tensor_component="trace",
                active_filter=viz.ActiveSubmodelPart("active", {1, 3}, set()),
                show_profile=True,
            )
            assert viewer.ax1 is not None
            assert viewer.profile_values is not None
            viewers.append(viewer)
    finally:
        for viewer in viewers:
            close_viewer_figures(viewer)


def test_interactive_viewer_update_plot_handles_gauss_point_types(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "gauss.post.res"
    write_mesh(mesh_path)
    result_path.write_text(
        'Result "GP_VECTOR" "Load" 1 Vector OnGaussPoints "GP"\n'
        "Values\n"
        "10 1.0 2.0\n"
        "End Values\n"
        'Result "GP_MATRIX" "Load" 1 Matrix OnGaussPoints "GP"\n'
        "Values\n"
        "10 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "End Values\n"
        'Result "GP_SCALAR" "Load" 1 Scalar OnGaussPoints "GP"\n'
        "Values\n"
        "10 7.0\n"
        "End Values\n",
        encoding="utf-8",
    )
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)

    viewers = []
    try:
        for result_name in ("GP_VECTOR", "GP_MATRIX", "GP_SCALAR"):
            viewer = viz.InteractiveViewer(
                headers=headers,
                res_path=result_path,
                all_nodes=all_nodes,
                triangles=triangles,
                surface_elements=surface_elements,
                line_meshes=line_meshes,
                profile_nodes=set(),
                initial_result=result_name,
                initial_step="latest",
                initial_location="OnGaussPoints",
                profile_x=0.0,
                x_tol=0.35,
                initial_tensor_component="trace",
                active_filter=(
                    None
                    if result_name == "GP_SCALAR"
                    else viz.ActiveSubmodelPart("active", set(), {10})
                ),
            )
            assert viewer.current_location == "OnGaussPoints"
            if result_name == "GP_SCALAR":
                assert viewer.ax0.collections[0].get_array().size == len(all_nodes)
            viewers.append(viewer)
    finally:
        for viewer in viewers:
            close_viewer_figures(viewer)


def test_interactive_viewer_update_plot_renders_profile_with_active_filter(tmp_path):
    viewer = make_single_viewer(
        tmp_path,
        profile_nodes={1, 3},
        active_filter=viz.ActiveSubmodelPart("active", {1, 3}, {10}),
        show_isolines=True,
        show_profile=True,
    )
    try:
        assert viewer.ax1 is not None
        assert viewer.profile_y_data is not None
        assert viewer.profile_values is not None
        viewer.show_gradient = True
        viewer.profile_x = float("nan")
        viewer.update_plot()
        assert viewer.profile_y_data is not None
        assert viewer.profile_values is not None
    finally:
        close_viewer_figures(viewer)


def test_resolve_active_nodes_handles_empty_and_comma_separated_selection():
    mesh_block_nodes = {
        "Surface": {1, 2, 3},
        "Profile": {3, 4},
    }

    assert viz.resolve_active_nodes(mesh_block_nodes, None) is None
    assert viz.resolve_active_nodes(mesh_block_nodes, "  ,  ") is None
    assert viz.resolve_active_nodes(mesh_block_nodes, " Surface, Profile ") == {1, 2, 3, 4}


def test_resolve_active_nodes_reports_unknown_names_and_available_blocks():
    mesh_block_nodes = {
        "Surface": {1, 2, 3},
        "Profile": {3, 4},
    }

    with pytest.raises(ValueError, match=r"Unknown --active submodelpart\(s\): Missing"):
        viz.resolve_active_nodes(mesh_block_nodes, "Surface, Missing")

    many_blocks = {f"Block_{index:02d}": {index} for index in range(25)}
    with pytest.raises(ValueError, match=r"\+5 more"):
        viz.resolve_active_nodes(many_blocks, "Unknown")


def test_scan_and_load_results_support_steps_and_result_types(tmp_path):
    result_path = tmp_path / "model.post.res"
    write_results(result_path)

    headers = viz.scan_result_headers(result_path)
    assert [(header.name, header.step, header.location) for header in headers] == [
        ("PRESSURE", 1, "OnNodes"),
        ("PRESSURE", 2, "OnNodes"),
        ("DISPLACEMENT", 2, "OnNodes"),
        ("STRESS", 2, "OnGaussPoints"),
    ]

    result_type, values = viz.load_values(result_path, "DISPLACEMENT", 2, "OnNodes")
    assert result_type == "Vector"
    np.testing.assert_allclose(values[1], [3.0, 4.0])


def test_pick_step_prefers_nodes_and_validates_requests():
    headers = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnGaussPoints", None, 1),
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnGaussPoints", None, 2),
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnNodes", None, 3),
    ]

    assert viz.pick_step(headers, "PRESSURE", "latest", None) == (2, "OnNodes")
    assert viz.pick_step(headers, "PRESSURE", "1.0", "OnGaussPoints") == (1, "OnGaussPoints")

    with pytest.raises(ValueError, match="Invalid step"):
        viz.pick_step(headers, "PRESSURE", "not-a-step", "OnNodes")
    with pytest.raises(ValueError, match="Step 4 not found"):
        viz.pick_step(headers, "PRESSURE", "4", "OnNodes")


def test_comparison_reports_common_and_unique_variables():
    first = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("FIRST_ONLY", 1, "Scalar", "OnNodes", None, 2),
    ]
    second = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("SECOND_ONLY", 1, "Scalar", "OnNodes", None, 2),
    ]

    assert viz.compare_result_names(first, second) == (
        ["PRESSURE"],
        ["FIRST_ONLY"],
        ["SECOND_ONLY"],
    )


def test_comparison_viewer_reset_and_adjust_scale():
    viewer = viz.ComparisonViewer.__new__(viz.ComparisonViewer)
    viewer.current_style = "background+isolines"
    viewer.fields = [
        {"scalar": np.array([1.0, np.nan])},
        {"scalar": np.array([-2.0, 4.0])},
    ]
    viewer.difference_field = {"scalar": np.array([-2.0, 5.0])}
    viewer.requested_vmin = 10.0
    viewer.requested_vmax = 20.0
    viewer.vmin = 10.0
    viewer.vmax = 20.0

    viewer._reset_scale()
    assert viewer.requested_vmin is None
    assert viewer.requested_vmax is None
    assert (viewer.vmin, viewer.vmax) == (-2.0, 4.0)

    viewer._adjust_scale(0.5)
    assert (viewer.requested_vmin, viewer.requested_vmax) == (-0.5, 2.5)
    assert (viewer.vmin, viewer.vmax) == (-0.5, 2.5)

    viewer.current_style = "difference"
    viewer._reset_scale()
    assert (viewer.vmin, viewer.vmax) == (-5.0, 5.0)


def test_shared_scale_uses_values_from_both_solutions():
    assert viz.calculate_shared_scale(
        [np.array([1.0, np.nan]), np.array([-2.0, 4.0])]
    ) == (-2.0, 4.0)
    assert viz.calculate_shared_scale(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0])], vmin=0.0, vmax=5.0
    ) == (0.0, 5.0)

    with pytest.raises(ValueError, match="minimum"):
        viz.calculate_shared_scale([np.array([1.0])], vmin=2.0, vmax=1.0)


def test_weighted_coordinate_inspection_uses_nearby_values(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    solution = viz.load_solution_data("model", "model", base_dir=tmp_path)
    values = {
        1: np.array([10.0]),
        2: np.array([20.0]),
        3: np.array([30.0]),
    }

    nearest = viz.ComparisonViewer._value_at_coordinate(solution, values, 0.5, 0.5, "nearest")
    weighted = viz.ComparisonViewer._value_at_coordinate(solution, values, 0.5, 0.5, "idw")

    assert float(nearest[3][0]) == 10.0
    assert float(weighted[3][0]) == 20.0


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


def test_get_available_steps_returns_sorted_unique_steps_for_current_result():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.current_result = "PRESSURE"
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("OTHER", 9, "Scalar", "OnNodes", None, 2),
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 3),
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 4),
    ]

    assert viewer.get_available_steps() == [1, 3]


def test_select_step_and_location_prefers_nodes_then_gauss_points():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 2),
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnGaussPoints", "GP", 3),
        viz.ResultHeader("PRESSURE", 4, "Scalar", "OnGaussPoints", "GP", 4),
    ]

    assert viewer._select_step_and_location_for_result("PRESSURE", 1) == (1, "OnNodes")
    assert viewer._select_step_and_location_for_result("PRESSURE", 2) == (2, "OnGaussPoints")
    assert viewer._select_step_and_location_for_result("PRESSURE", 8) == (3, "OnNodes")


def test_select_step_and_location_falls_back_to_latest_gauss_point_or_errors():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnGaussPoints", "GP", 1),
        viz.ResultHeader("PRESSURE", 5, "Scalar", "OnGaussPoints", "GP", 2),
    ]

    assert viewer._select_step_and_location_for_result("PRESSURE", 8) == (5, "OnGaussPoints")
    with pytest.raises(ValueError, match="No available steps for result 'MISSING'"):
        viewer._select_step_and_location_for_result("MISSING", 1)


def test_format_result_line_handles_empty_scalar_vector_and_matrix_values():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)

    assert viewer._format_result_line("EMPTY", 1, "OnNodes", "Scalar", np.array([])) == (
        "EMPTY: <empty> [step 1, OnNodes]"
    )
    assert viewer._format_result_line("PRESSURE", 2, "OnNodes", "Scalar", np.array([12.3456789])) == (
        "PRESSURE: 12.3457 [step 2, OnNodes]"
    )

    vector_line = viewer._format_result_line(
        "DISPLACEMENT", 3, "OnNodes", "Vector", np.array([3.0, 4.0])
    )
    assert vector_line == (
        "DISPLACEMENT: x=3, y=4, z=nan, |v|=5 [step 3, OnNodes]"
    )

    matrix_line = viewer._format_result_line(
        "STRESS", 4, "OnGaussPoints", "Matrix", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )
    assert matrix_line == (
        "STRESS: xx=1, xy=4, yx=4, yy=2, trace=6, |T|=9.53939 "
        "[step 4, OnGaussPoints]"
    )


def test_build_node_report_formats_values_missing_nodes_and_errors():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.result_names = ["PRESSURE", "DISPLACEMENT", "STRESS", "MISSING", "BROKEN"]
    viewer.headers = [viz.ResultHeader("PRESSURE", 3, "Scalar", "OnNodes", None, 1)]
    viewer.get_current_step = lambda: 2
    viewer._select_step_and_location_for_result = lambda name, step: (step, "OnNodes")

    result_data = {
        "PRESSURE": ("Scalar", {7: np.array([12.5])}),
        "DISPLACEMENT": ("Vector", {7: np.array([3.0, 4.0])}),
        "STRESS": ("Matrix", {7: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])}),
        "MISSING": ("Scalar", {}),
    }

    def load_cached_values(name, step, location):
        if name == "BROKEN":
            raise RuntimeError("load failed")
        return result_data[name]

    viewer._load_cached_values = load_cached_values
    report = viewer._build_node_report(1.25, 2.5, 7, 1.0, 2.0, 0.25)

    assert "Mouse click: x=1.25, y=2.5" in report
    assert "Nearest node: id=7, x=1, y=2" in report
    assert "PRESSURE: 12.5 [step 2, OnNodes]" in report
    assert "DISPLACEMENT: x=3, y=4, z=nan, |v|=5 [step 2, OnNodes]" in report
    assert "STRESS: xx=1, xy=4, yx=4, yy=2, trace=6, |T|=9.53939 [step 2, OnNodes]" in report
    assert "MISSING: <no value> [step 2, OnNodes]" in report
    assert "BROKEN: <error: load failed>" in report


def test_build_node_report_uses_latest_header_step_when_current_step_fails():
    viewer = viz.InteractiveViewer.__new__(viz.InteractiveViewer)
    viewer.result_names = ["PRESSURE"]
    viewer.headers = [
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("PRESSURE", 5, "Scalar", "OnNodes", None, 2),
    ]
    viewer.get_current_step = lambda: (_ for _ in ()).throw(RuntimeError("no current step"))
    selected = []

    def select_step(name, preferred_step):
        selected.append(preferred_step)
        return preferred_step, "OnNodes"

    viewer._select_step_and_location_for_result = select_step
    viewer._load_cached_values = lambda name, step, location: ("Scalar", {1: np.array([4.0])})

    report = viewer._build_node_report(0.0, 0.0, 1, 0.0, 0.0, 0.0)

    assert selected == [5]
    assert "PRESSURE: 4 [step 5, OnNodes]" in report


def make_single_viewer(tmp_path, **overrides):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)
    options = {
        "headers": headers,
        "res_path": result_path,
        "all_nodes": all_nodes,
        "triangles": triangles,
        "surface_elements": surface_elements,
        "line_meshes": line_meshes,
        "profile_nodes": set(),
        "initial_result": "PRESSURE",
        "initial_step": "latest",
        "initial_location": "OnNodes",
        "profile_x": 0.0,
        "x_tol": 0.35,
        "initial_tensor_component": "trace",
        "active_filter": None,
    }
    options.update(overrides)
    return viz.InteractiveViewer(**options)


def close_viewer_figures(viewer):
    if getattr(viewer, "animation_running", False):
        viewer._stop_animation()
    for attribute in (
        "coord_input_fig",
        "info_fig",
        "scale_input_fig",
        "deformation_input_fig",
        "fig",
    ):
        figure = getattr(viewer, attribute, None)
        if figure is not None:
            viz.plt.close(figure)


def test_interactive_viewer_info_popup_and_markers(tmp_path):
    viewer = make_single_viewer(tmp_path)
    try:
        viewer._show_info_popup("inspection text")
        assert viewer.info_fig is not None
        viewer._show_info_popup("updated inspection text")

        viewer._inspect_coordinate(0.1, 0.1, source="test")
        assert len(viewer._click_markers) == 2
        viewer._clear_click_markers()
        assert viewer._click_markers == []
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_menu_drag_handlers(tmp_path):
    viewer = make_single_viewer(tmp_path)
    try:
        menu_artist = viewer.menu_artists[0]
        original_position = menu_artist.get_position()
        viewer._on_menu_pick(SimpleNamespace(
            artist=menu_artist,
            mouseevent=SimpleNamespace(x=100, y=100),
        ))
        assert viewer.menu_drag_artist is menu_artist
        assert viewer.menu_drag_offset is not None

        viewer._on_menu_motion(SimpleNamespace(x=180, y=160))
        assert menu_artist.get_position() != original_position

        viewer._on_menu_release(SimpleNamespace())
        assert viewer.menu_drag_artist is None
        assert viewer.menu_drag_offset is None
        dragged_position = menu_artist.get_position()
        viewer.update_plot()
        assert viewer.menu_artists[0].get_position() == dragged_position
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_coordinate_input_and_clicks(tmp_path):
    viewer = make_single_viewer(tmp_path)
    try:
        viewer._open_coordinate_input_popup()
        assert viewer.coord_input_fig is not None
        assert viewer.coord_textbox is not None
        assert viewer.coord_status_artist is not None

        inspected = []
        viewer._inspect_coordinate = lambda x, y, source="mouse": inspected.append((x, y, source))
        viewer.coord_textbox._observers.process("submit", "invalid")
        assert "exactly two numbers" in viewer.coord_status_artist.get_text()
        viewer.coord_textbox._observers.process("submit", "x, y")
        assert "Invalid numeric format" in viewer.coord_status_artist.get_text()
        viewer.coord_textbox._observers.process("submit", "0.75, 1.25")
        assert inspected == [(0.75, 1.25, "typed")]

        viewer.on_click(SimpleNamespace(inaxes=viewer.ax0, key=None, button=1, xdata=0.25, ydata=0.5))
        assert inspected == [(0.75, 1.25, "typed"), (0.25, 0.5, "mouse")]
        for event in (
            SimpleNamespace(inaxes=viewer.ax0, key="shift", button=1, xdata=0.25, ydata=0.5),
            SimpleNamespace(inaxes=viewer.ax0, key=None, button=3, xdata=0.25, ydata=0.5),
            SimpleNamespace(inaxes=viewer.ax0, key=None, button=1, xdata=None, ydata=0.5),
            SimpleNamespace(inaxes=None, key=None, button=1, xdata=0.25, ydata=0.5),
        ):
            viewer.on_click(event)
        assert inspected == [(0.75, 1.25, "typed"), (0.25, 0.5, "mouse")]
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_zoom_pan_scroll(tmp_path):
    viewer = make_single_viewer(tmp_path)
    try:
        viewer.ax0.set_xlim(0.0, 2.0)
        viewer.ax0.set_ylim(0.0, 4.0)
        viewer._store_current_view()
        assert viewer.view_xlim == (0.0, 2.0)
        assert viewer.view_ylim == (0.0, 4.0)
        viewer._zoom_main_axes(0.5, center_x=1.0, center_y=2.0)
        assert viewer.ax0.get_xlim() == (0.5, 1.5)
        assert viewer.ax0.get_ylim() == (1.0, 3.0)
        saved_xlim = viewer.ax0.get_xlim()
        viewer._zoom_main_axes(0.0)
        assert viewer.ax0.get_xlim() == saved_xlim
        viewer.on_button_press(SimpleNamespace(inaxes=viewer.ax0, button=3, key=None, xdata=1.0, ydata=2.0))
        assert viewer._pan_active
        viewer.on_mouse_move(SimpleNamespace(inaxes=viewer.ax0, xdata=1.2, ydata=2.5))
        np.testing.assert_allclose(viewer.ax0.get_xlim(), (0.3, 1.3))
        np.testing.assert_allclose(viewer.ax0.get_ylim(), (0.5, 2.5))
        viewer.on_button_release(SimpleNamespace())
        assert not viewer._pan_active
        old_width = viewer.ax0.get_xlim()[1] - viewer.ax0.get_xlim()[0]
        viewer.on_scroll(SimpleNamespace(inaxes=viewer.ax0, button="up", xdata=0.8, ydata=1.5))
        assert viewer.ax0.get_xlim()[1] - viewer.ax0.get_xlim()[0] < old_width
        new_width = viewer.ax0.get_xlim()[1] - viewer.ax0.get_xlim()[0]
        viewer.on_scroll(SimpleNamespace(inaxes=viewer.ax0, button="down", xdata=0.8, ydata=1.5))
        assert viewer.ax0.get_xlim()[1] - viewer.ax0.get_xlim()[0] > new_width
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_profile_export(tmp_path, monkeypatch, capsys):
    viewer = make_single_viewer(tmp_path)
    try:
        monkeypatch.chdir(tmp_path)
        viewer.export_profile_data()
        assert "No profile data to export" in capsys.readouterr().out
        viewer._set_profile_data([0.0, 1.0], [2.0, 4.0], profile_x_current=None)
        viewer.export_profile_data()
        output_path = tmp_path / "PRESSURE_latest_OnNodes_magnitude_profile.csv"
        assert output_path.is_file()
        assert output_path.read_text().splitlines() == [
            "y_coordinate,value,gradient", "0.0,2.0,2.0", "1.0,4.0,2.0",
        ]
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_renders_deformed_mesh_with_typed_magnification(tmp_path):
    viewer = make_single_viewer(
        tmp_path,
        initial_result="DISPLACEMENT",
        initial_show_deformation=True,
        initial_deformation_scale=2.0,
    )
    try:
        legend_labels = [text.get_text() for text in viewer.ax0.get_legend().get_texts()]
        assert "Deformed mesh (x2)" in legend_labels

        viewer._open_deformation_scale_popup()
        viewer.deformation_textbox._observers.process("submit", "invalid")
        assert "positive finite number" in viewer.deformation_status_artist.get_text()
        viewer.deformation_textbox._observers.process("submit", "3.5")
        assert viewer.deformation_scale == 3.5
        assert viewer.show_deformation
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_does_not_deform_reaction_results(tmp_path):
    result_path = tmp_path / "model.post.res"
    write_mesh(tmp_path / "model.post.msh")
    write_results(result_path)
    with result_path.open("a", encoding="utf-8") as result_file:
        result_file.write(
            'Result "REACTION" "Load" 1 Vector OnNodes\n'
            "Values\n"
            "1 3.0 4.0\n"
            "2 0.0 2.0\n"
            "3 1.0 1.0\n"
            "End Values\n"
        )

    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(
        tmp_path / "model.post.msh"
    )
    viewer = viz.InteractiveViewer(
        headers=viz.scan_result_headers(result_path),
        res_path=result_path,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements,
        line_meshes=line_meshes,
        profile_nodes=set(),
        initial_result="REACTION",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
        initial_show_deformation=True,
    )
    try:
        assert viewer._is_displacement_result() is False
        assert viewer.show_deformation is False
        viewer.on_key(SimpleNamespace(key="u"))
        assert viewer.show_deformation is False
        assert viewer.ax0.get_legend() is None

        viewer._open_deformation_scale_popup()
        viewer.deformation_textbox._observers.process("submit", "2")
        assert "requires a DISPLACEMENT" in viewer.deformation_status_artist.get_text()
    finally:
        close_viewer_figures(viewer)


def test_export_current_field_data_and_save_current_plot(tmp_path, monkeypatch, capsys):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)
    viewer = viz.InteractiveViewer(
        headers=headers,
        res_path=result_path,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements,
        line_meshes=line_meshes,
        profile_nodes=set(),
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
    )
    try:
        viewer.export_current_field_data()
        assert "only available while gradient mode is on" in capsys.readouterr().out

        viewer.show_gradient = True
        viewer.export_current_field_data()
        assert "No gradient field is currently available" in capsys.readouterr().out

        viewer.current_field_node_ids = np.array([1, 2])
        viewer.current_field_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        viewer.current_field_source_values = np.array([1.0, np.nan])
        viewer.current_field_display_values = np.array([1.0, 2.0])
        viewer.current_field_grad_x = np.array([2.0, np.nan])
        viewer.current_field_grad_y = np.array([3.0, np.nan])
        viewer.current_field_grad_magnitude = np.array([np.sqrt(13.0), np.nan])
        monkeypatch.chdir(tmp_path)
        viewer.export_current_field_data()
        output_path = tmp_path / "PRESSURE_latest_OnNodes_magnitude_gradient_nodes.csv"
        assert output_path.is_file()
        assert len(output_path.read_text().splitlines()) == 2

        saved_path = viewer.save_current_plot(tmp_path / "current_plot.png")
        assert saved_path.is_file()
        default_saved_path = viewer.save_current_plot()
        assert default_saved_path.resolve() == tmp_path / "PRESSURE_latest_OnNodes_interactive.png"
        assert default_saved_path.is_file()
        settings_path = viewer.save_current_settings(tmp_path / "single_settings.json")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["mode"] == "single"
        assert settings["result"] == "PRESSURE"
        assert settings["show_gradient"]
    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_on_key_dispatches_actions(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    headers = viz.scan_result_headers(result_path)
    viewer = viz.InteractiveViewer(
        headers=headers,
        res_path=result_path,
        all_nodes=all_nodes,
        triangles=triangles,
        surface_elements=surface_elements,
        line_meshes=line_meshes,
        profile_nodes=set(),
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
    )
    calls = []
    viewer.update_plot = lambda: calls.append("update")
    viewer.save_current_plot = lambda: calls.append("save")
    viewer.export_profile_data = lambda: calls.append("profile")
    viewer.export_current_field_data = lambda: calls.append("gradient")
    viewer._open_coordinate_input_popup = lambda: calls.append("coordinate")
    try:
        for key in ("n", "p", "k", "j", "x", "y", "z", "g", "t", "d", "m", "l", "r", "+", "-", "c"):
            viewer.on_key(SimpleNamespace(key=key))
        viewer.on_key(SimpleNamespace(key="s"))
        viewer.on_key(SimpleNamespace(key="e"))
        viewer.on_key(SimpleNamespace(key="E"))
        viewer.on_key(SimpleNamespace(key="i"))
        viewer.on_key(SimpleNamespace(key="?"))
        assert "save" in calls
        assert "profile" in calls
        assert "gradient" in calls
        assert "coordinate" in calls
        assert calls.count("update") == 13
    finally:
        close_viewer_figures(viewer)


def test_render_plot_static_handles_scalar_vector_matrix_and_filters(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    all_nodes, triangles, surface_elements, line_meshes, _ = viz.parse_post_msh(mesh_path)
    profile_nodes = {1, 3}

    _, scalar_values = viz.load_values(result_path, "PRESSURE", 2, "OnNodes")
    scalar_path = tmp_path / "scalar.png"
    viz.render_plot_static(
        all_nodes=all_nodes,
        triangles=triangles,
        values=scalar_values,
        result_name="PRESSURE",
        step=2,
        result_type="Scalar",
        component="magnitude",
        tensor_component_mode="trace",
        profile_nodes=profile_nodes,
        out_path=scalar_path,
        active_filter=None,
        show_isolines=True,
        show_profile=True,
    )
    assert scalar_path.is_file()

    vector_type, vector_values = viz.load_values(result_path, "DISPLACEMENT", 2, "OnNodes")
    vector_path = tmp_path / "vector.png"
    viz.render_plot_static(
        all_nodes, triangles, vector_values, "DISPLACEMENT", 2, vector_type,
        "magnitude", "trace", profile_nodes, vector_path, None,
    )
    assert vector_path.is_file()

    matrix_type, matrix_values = viz.load_values(result_path, "STRESS", 2, "OnGaussPoints")
    line_elements = [element for _, _, elements in line_meshes for element in elements]
    matrix_values = viz.project_element_to_nodes(
        matrix_values, surface_elements, line_elements
    )
    matrix_path = tmp_path / "matrix.png"
    viz.render_plot_static(
        all_nodes, triangles, matrix_values, "STRESS", 2, matrix_type,
        "magnitude", "trace", profile_nodes, matrix_path, None,
    )
    assert matrix_path.is_file()

    filtered_path = tmp_path / "filtered.png"
    viz.render_plot_static(
        all_nodes, triangles, scalar_values, "PRESSURE", 2, "Scalar",
        "magnitude", "trace", profile_nodes, filtered_path,
        viz.ActiveSubmodelPart("active", {1, 3}, set()),
    )
    assert filtered_path.is_file()


def test_render_plot_interactive_constructs_viewer_and_shows(monkeypatch):
    captured = {}

    class FakeViewer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(viz, "InteractiveViewer", FakeViewer)
    shown = []
    monkeypatch.setattr(viz.plt, "show", lambda: shown.append(True))

    viz.render_plot_interactive(
        headers=[],
        res_path=viz.Path("results.post.res"),
        all_nodes={},
        triangles=[],
        surface_elements=[],
        line_meshes=[],
        profile_nodes=set(),
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
        show_isolines=True,
        show_profile=True,
    )

    assert captured["initial_result"] == "PRESSURE"
    assert captured["show_isolines"]
    assert captured["show_profile"]
    assert shown == [True]


def test_print_result_summary_reports_grouped_results(capsys):
    headers = [
        viz.ResultHeader("PRESSURE", 1, "Scalar", "OnNodes", None, 1),
        viz.ResultHeader("PRESSURE", 2, "Scalar", "OnNodes", None, 2),
        viz.ResultHeader("STRESS", 3, "Matrix", "OnGaussPoints", "GP", 3),
    ]

    viz.print_result_summary(headers)

    output = capsys.readouterr().out
    assert "Available results:" in output
    assert "PRESSURE" in output
    assert "steps=[1..2], count=2" in output
    assert "STRESS" in output
    assert "gp=GP" in output


def test_print_comparison_summary_reports_common_and_unique_results(tmp_path, capsys):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)

    viz.print_comparison_summary(first, second)

    output = capsys.readouterr().out
    assert "Common variables (first and second):" in output
    assert "DISPLACEMENT" in output
    assert "Only in first" in output
    assert "STRESS" in output
    assert "Only in second" in output
    assert "TEMPERATURE" in output


def test_main_lists_single_solution_results(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["visualize_gid_results.py", "model", "--list"])

    viz.main()

    assert "Available results:" in capsys.readouterr().out


def test_main_dispatches_single_static_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_plot_static", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "model", "--result", "PRESSURE", "--save", "output.png", "--isolines", "--profile"],
    )

    viz.main()

    assert captured["result_name"] == "PRESSURE"
    assert captured["show_isolines"]
    assert captured["show_profile"]


def test_main_dispatches_single_interactive_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_plot_interactive", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "model", "--isolines", "--profile"],
    )

    viz.main()

    assert captured["initial_result"] == "WATER_PRESSURE"
    assert captured["show_isolines"]
    assert captured["show_profile"]


def test_discover_solution_paths_finds_matching_pairs(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")

    assert viz.discover_solution_paths(tmp_path) == [
        ((tmp_path / "first.post.msh").resolve(), (tmp_path / "first.post.res").resolve()),
        ((tmp_path / "second.post.msh").resolve(), (tmp_path / "second.post.res").resolve()),
    ]


def test_discover_solution_paths_orders_numeric_stage_then_alpha(tmp_path):
    for name in (
        "10_eemdijk",
        "2_eemdijk",
        "case_stage10",
        "case_stage2",
        "zeta",
        "Alpha",
    ):
        (tmp_path / f"{name}.post.msh").touch()
        (tmp_path / f"{name}.post.res").touch()

    pairs = viz.discover_solution_paths(tmp_path)

    assert [path[0].stem for path in pairs] == [
        "2_eemdijk.post",
        "10_eemdijk.post",
        "case_stage2.post",
        "case_stage10.post",
        "Alpha.post",
        "zeta.post",
    ]


def test_interactive_viewer_animates_all_solutions_and_steps(tmp_path, monkeypatch):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    with (tmp_path / "second.post.res").open("a", encoding="utf-8") as result_file:
        result_file.write(
            'Result "PRESSURE" "OtherLoad" 2 Scalar OnNodes\n'
            "Values\n"
            "101 30.0\n"
            "102 31.0\n"
            "103 32.0\n"
            "End Values\n"
        )
    solution_paths = viz.discover_solution_paths(tmp_path)
    first = viz.load_solution_data(str(solution_paths[0][0]), "first")
    viewer = viz.InteractiveViewer(
        headers=first.headers,
        res_path=first.res_path,
        all_nodes=first.all_nodes,
        triangles=first.triangles,
        surface_elements=first.surface_elements,
        line_meshes=first.line_meshes,
        profile_nodes=first.profile_nodes,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
        solution_paths=solution_paths,
    )
    try:
        assert viewer._build_animation_frames() == [
            (0, "PRESSURE", "1"),
            (0, "PRESSURE", "2"),
            (1, "PRESSURE", "1"),
            (1, "PRESSURE", "2"),
        ]

        class FakeEventSource:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

        class FakeAnimation:
            def __init__(self):
                self.event_source = FakeEventSource()

        fake_animation = FakeAnimation()
        monkeypatch.setattr(viz.animation, "FuncAnimation", lambda *args, **kwargs: fake_animation)
        viewer._toggle_animation()
        assert viewer.animation_running
        viewer._toggle_animation()
        assert not viewer.animation_running
        assert fake_animation.event_source.stopped

        viewer._advance_animation_frame(2)
        assert viewer.solution_index == 1
        assert viewer.current_step_str == "1"

        viewer._advance_animation_frame(0)
        assert viewer.solution_index == 0
        assert viewer.current_step_str == "1"

    finally:
        close_viewer_figures(viewer)


def test_interactive_viewer_switches_folder_solutions(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    solution_paths = viz.discover_solution_paths(tmp_path)
    first = viz.load_solution_data(str(solution_paths[0][0]), "first")
    viewer = viz.InteractiveViewer(
        headers=first.headers,
        res_path=first.res_path,
        all_nodes=first.all_nodes,
        triangles=first.triangles,
        surface_elements=first.surface_elements,
        line_meshes=first.line_meshes,
        profile_nodes=first.profile_nodes,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=0.0,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
        solution_paths=solution_paths,
    )
    try:
        assert viewer.solution_index == 0
        assert viewer.ax0.get_xlim() == (0.0, 2.0)
        viewer.current_step_str = "3369600"
        viewer.on_key(SimpleNamespace(key="N"))
        assert viewer.solution_index == 1
        assert viewer.res_path == solution_paths[1][1]
        assert viewer.open_file_name == "second.post.res"
        assert viewer.ax0.get_xlim() == (0.0, 2.0)
        assert viewer.current_step_str == "latest"
        assert not any(text.get_text().startswith("Error:") for text in viewer.ax0.texts)
        assert viewer.current_result == "PRESSURE"
        viewer.on_key(SimpleNamespace(key="P"))
        assert viewer.solution_index == 0
        assert viewer.res_path == solution_paths[0][1]
        assert viewer.open_file_name == "first.post.res"
        assert viewer.ax0.get_xlim() == (0.0, 2.0)
        for _ in range(3):
            viewer.on_key(SimpleNamespace(key="N"))
            assert viewer.ax0.get_xlim() == (0.0, 2.0)
            viewer.on_key(SimpleNamespace(key="P"))
            assert viewer.ax0.get_xlim() == (0.0, 2.0)
    finally:
        close_viewer_figures(viewer)


def write_grid_stage(path, with_vertical_lines):
    coordinates = "".join(
        f"{1 + row * 3 + col} {col * 1.5} {row * 1.5} 0.0\n"
        for row in range(3)
        for col in range(3)
    )
    text = (
        'MESH "Soil" dimension 2 ElemType Triangle Nnode 3\n'
        f"Coordinates\n{coordinates}End Coordinates\n"
        "Elements\n"
        "1 1 2 5 1\n2 1 5 4 1\n3 2 3 6 1\n4 2 6 5 1\n"
        "5 4 5 8 1\n6 4 8 7 1\n7 5 6 9 1\n8 5 9 8 1\n"
        "End Elements\n"
    )
    if with_vertical_lines:
        for name, (bottom, middle, top), first_element in (
            ("Left", (1, 4, 7), 20),
            ("Right", (3, 6, 9), 30),
        ):
            text += (
                f'MESH "{name}" dimension 2 ElemType Linear Nnode 2\n'
                "Coordinates\nEnd Coordinates\n"
                "Elements\n"
                f"{first_element} {bottom} {middle} 1\n"
                f"{first_element + 1} {middle} {top} 1\n"
                "End Elements\n"
            )
    path.with_suffix(".msh").write_text(text, encoding="utf-8")
    path.with_suffix(".res").write_text(
        'Result "PRESSURE" "Load" 1 Scalar OnNodes\nValues\n'
        + "".join(f"{node_id} {float(node_id)}\n" for node_id in range(1, 10))
        + "End Values\n",
        encoding="utf-8",
    )


def test_interactive_viewer_keeps_requested_profile_x_when_switching_folder_solutions(tmp_path):
    write_grid_stage(tmp_path / "1_without_wall.post", with_vertical_lines=False)
    write_grid_stage(tmp_path / "2_with_walls.post", with_vertical_lines=True)
    solution_paths = viz.discover_solution_paths(tmp_path)
    first = viz.load_solution_data(str(solution_paths[0][0]), "first")
    profile_x = 1.5
    viewer = viz.InteractiveViewer(
        headers=first.headers,
        res_path=first.res_path,
        all_nodes=first.all_nodes,
        triangles=first.triangles,
        surface_elements=first.surface_elements,
        line_meshes=first.line_meshes,
        profile_nodes=viz.detect_profile_nodes(
            first.all_nodes, first.line_meshes, profile_x=profile_x, x_tol=0.35
        ),
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location="OnNodes",
        profile_x=profile_x,
        x_tol=0.35,
        initial_tensor_component="trace",
        active_filter=None,
        solution_paths=solution_paths,
        show_profile=True,
    )

    def profile_node_x_coordinates():
        return {viewer.all_nodes[node_id][0] for node_id in viewer.profile_nodes}

    try:
        assert profile_node_x_coordinates() == {profile_x}
        viewer.on_key(SimpleNamespace(key="N"))
        assert viewer.solution_index == 1
        assert profile_node_x_coordinates() == {profile_x}
        viewer.on_key(SimpleNamespace(key="P"))
        assert viewer.solution_index == 0
        assert profile_node_x_coordinates() == {profile_x}
    finally:
        close_viewer_figures(viewer)


def test_main_dispatches_folder_interactive_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    captured = {}
    monkeypatch.setattr(viz, "render_plot_interactive", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--folder", str(tmp_path), "--result", "PRESSURE"],
    )

    viz.main()

    assert len(captured["solution_paths"]) == 2
    assert captured["solution_paths"][0][0].name == "first.post.msh"


def test_main_loads_single_visualization_settings(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    settings_path = tmp_path / "single_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "mode": "single",
                "result": "PRESSURE",
                "step": "1",
                "location": "OnNodes",
                "component": "x",
                "tensor_component": "trace",
                "profile_x": 0.25,
                "x_tol": 0.1,
                "show_gradient": True,
                "show_mesh": True,
                "show_isolines": True,
                "show_profile": True,
                "show_isoline_labels": True,
                "colormap": "plasma",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_plot_interactive", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "model", "--settings", str(settings_path)],
    )

    viz.main()

    assert captured["initial_result"] == "PRESSURE"
    assert captured["initial_step"] == "1"
    assert captured["initial_component"] == "x"
    assert captured["initial_show_gradient"]
    assert captured["initial_show_mesh"]
    assert captured["initial_colormap"] == "plasma"
    assert captured["show_profile"]


def test_main_loads_comparison_visualization_settings(tmp_path, monkeypatch):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    settings_path = tmp_path / "comparison_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "mode": "comparison",
                "solutions": [
                    {"mesh": str(tmp_path / "first.post.msh")},
                    {"mesh": str(tmp_path / "second.post.msh")},
                ],
                "result": "DISPLACEMENT",
                "step": "latest",
                "location": "OnNodes",
                "component": "magnitude",
                "tensor_component": "trace",
                "profile_x": 0.0,
                "style": "difference",
                "inspection_method": "idw",
                "colormap": "cividis",
                "show_profile": True,
                "show_domain_edge": False,
                "vmin": -1.0,
                "vmax": 2.0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_comparison", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--settings", str(settings_path)],
    )

    viz.main()

    assert captured["initial_result"] == "DISPLACEMENT"
    assert captured["style"] == "difference"
    assert captured["inspection_method"] == "idw"
    assert captured["colormap"] == "cividis"
    assert captured["show_profile"]
    assert not captured["show_domain_edge"]
    assert captured["vmin"] == -1.0
    assert captured["vmax"] == 2.0


def test_main_dispatches_comparison_and_replaces_non_common_result(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_render(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viz, "render_comparison", fake_render)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "first", "--compare", "second",
            "--result", "TEMPERATURE", "--comparison-style", "difference",
            "--inspection-method", "idw", "--colormap", "cividis",
        ],
    )

    viz.main()

    assert captured["initial_result"] == "DISPLACEMENT"
    assert captured["style"] == "difference"
    assert captured["inspection_method"] == "idw"
    assert captured["colormap"] == "cividis"
    assert "not common" in capsys.readouterr().out


def test_main_dispatches_extract_json(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "extract.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_export(json_path, csv_out=None):
        captured["json_path"] = json_path
        captured["csv_out"] = csv_out
        return tmp_path / "output.csv"

    monkeypatch.setattr(viz, "export_line_csv_from_json", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--extract-json", str(config_path), "--csv-out", "out.csv"],
    )

    viz.main()

    assert captured["json_path"] == config_path.resolve()
    assert captured["csv_out"] == (tmp_path / "out.csv").resolve()
    assert "Saved CSV" in capsys.readouterr().out


@pytest.mark.parametrize("show", [False, True])
def test_main_plots_the_extracted_csv_with_plot_saved_results(tmp_path, monkeypatch, show):
    import plot_saved_results

    extract_path = tmp_path / "extract.json"
    extract_path.write_text("{}", encoding="utf-8")
    plot_config_path = tmp_path / "plots.json"
    plot_config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "export_line_csv_from_json", lambda json_path, csv_out=None: tmp_path / "out.csv")
    plot_calls = []
    monkeypatch.setattr(plot_saved_results, "main", lambda argv: plot_calls.append(argv) or 0)
    argv = ["visualize_gid_results.py", "--extract-json", str(extract_path), "--plot-config", str(plot_config_path)]
    if show:
        argv.append("--plot-show")
    monkeypatch.setattr(sys, "argv", argv)

    viz.main()

    expected = [str(plot_config_path.resolve())] + (["--show"] if show else [])
    assert plot_calls == [expected]


def test_main_reports_a_failing_plot_config(tmp_path, monkeypatch):
    import plot_saved_results

    extract_path = tmp_path / "extract.json"
    extract_path.write_text("{}", encoding="utf-8")
    plot_config_path = tmp_path / "plots.json"
    plot_config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "export_line_csv_from_json", lambda json_path, csv_out=None: tmp_path / "out.csv")
    monkeypatch.setattr(plot_saved_results, "main", lambda argv: 1)
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--extract-json", str(extract_path), "--plot-config", str(plot_config_path)],
    )

    with pytest.raises(RuntimeError, match="Plotting with"):
        viz.main()


def test_main_reports_a_missing_plot_config(tmp_path, monkeypatch):
    extract_path = tmp_path / "extract.json"
    extract_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(viz, "export_line_csv_from_json", lambda json_path, csv_out=None: tmp_path / "out.csv")
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_gid_results.py", "--extract-json", str(extract_path), "--plot-config", str(tmp_path / "missing.json")],
    )

    with pytest.raises(FileNotFoundError, match="Plot config not found"):
        viz.main()


@pytest.mark.parametrize(
    "extra_args",
    [["model", "--plot-config", "plots.json"], ["--extract-json", "extract.json", "--plot-show"]],
)
def test_main_rejects_plot_options_without_their_prerequisites(monkeypatch, extra_args):
    monkeypatch.setattr(sys, "argv", ["visualize_gid_results.py", *extra_args])

    with pytest.raises(SystemExit):
        viz.main()


def test_main_uses_explicit_active_mdpa_filter(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    mdpa_path = tmp_path / "active.mdpa"
    write_active_mdpa(mdpa_path)
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "PRESSURE",
            "--mdpa", str(mdpa_path), "--active", "active", "--save", "out.png",
        ],
    )

    viz.main()

    assert captured["active_filter"].name == "active"
    assert captured["active_filter"].node_ids == {1, 3}


def test_main_uses_automatic_active_mdpa_and_handles_empty_filter(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    write_active_mdpa(tmp_path / "model.mdpa", submodel_name="other")
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "PRESSURE",
            "--active", "missing", "--save", "out.png",
        ],
    )

    viz.main()

    assert captured["active_filter"] is None
    assert "No active filtering will be applied" in capsys.readouterr().out


def test_main_active_filter_requires_an_automatically_located_mdpa(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "PRESSURE",
            "--active", "active", "--save", "out.png",
        ],
    )

    with pytest.raises(FileNotFoundError, match="Could not auto-locate"):
        viz.main()


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


def test_export_line_csv_handles_vector_results(tmp_path):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    config_path = tmp_path / "vector_extract.json"
    config_path.write_text(
        '{"files": ["model"], "result": "DISPLACEMENT", "component": "y", '
        '"line_points": [[0, 0], [0, 1]], "line_samples": 2}',
        encoding="utf-8",
    )

    output_path = viz.export_line_csv_from_json(config_path)

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert [float(row["value"]) for row in rows] == pytest.approx([4.0, 3.333333333333111])
    assert [row["component"] for row in rows] == ["y", "y"]

    gauss_config_path = tmp_path / "gauss_extract.json"
    gauss_config_path.write_text(
        '{"files": ["model"], "result": "STRESS", "location": "OnGaussPoints", '
        '"line": {"start": [0, 0], "end": [0, 1], "samples": 2}}',
        encoding="utf-8",
    )
    gauss_output_path = viz.export_line_csv_from_json(gauss_config_path)
    gauss_rows = list(csv.DictReader(gauss_output_path.open(newline="", encoding="utf-8")))
    assert len(gauss_rows) == 2
    assert [float(row["value"]) for row in gauss_rows] == pytest.approx([1.0, 1.0])


def test_main_static_projects_gauss_point_results(tmp_path, monkeypatch, capsys):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "STRESS",
            "--location", "OnGaussPoints", "--save", "stress.png",
        ],
    )

    viz.main()

    assert len(captured["values"]) == 3
    assert "Projecting STRESS" in capsys.readouterr().out


def test_main_static_rejects_empty_result_values(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    (tmp_path / "model.post.res").write_text(
        'Result "EMPTY" "Load" 1 Scalar OnNodes\nEnd Values\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--result", "EMPTY",
            "--save", "empty.png",
        ],
    )

    with pytest.raises(RuntimeError, match="Could not load values"):
        viz.main()


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


def test_main_passes_profile_path_to_static_render(tmp_path, monkeypatch):
    write_mesh(tmp_path / "model.post.msh")
    write_results(tmp_path / "model.post.res")
    monkeypatch.chdir(tmp_path)
    captured = {}
    monkeypatch.setattr(viz, "render_plot_static", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_gid_results.py", "model", "--save", "profile.png",
            "--result", "PRESSURE", "--profile", "--profile-path", "0,0;0.5,0;0.5,0.5",
        ],
    )

    viz.main()

    assert captured["profile_path"] == [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]


def test_interactive_viewer_renders_profile_path(tmp_path):
    viewer = make_single_viewer(
        tmp_path,
        show_profile=True,
        profile_path=[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
    )
    try:
        assert viewer.ax1 is not None
        assert viewer.ax1.get_title() == "Profile path"
        assert viewer.profile_y_data is not None
        assert viewer.profile_values is not None
        assert viewer.profile_y_data[-1] == pytest.approx(0.5 + np.sqrt(0.5))
    finally:
        close_viewer_figures(viewer)


def test_draw_scalar_isolines_returns_solid_contours():
    figure, axis = viz.plt.subplots()
    try:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        tri = viz.mtri.Triangulation(points[:, 0], points[:, 1], np.array([[0, 1, 2]]))
        contours = viz.draw_scalar_isolines(axis, tri, np.array([0.0, 1.0, 2.0]), 0.0, 2.0)
        assert contours is not None
        assert all(style[1] is None for style in contours.get_linestyles())
    finally:
        viz.plt.close(figure)


def test_draw_scalar_isolines_skips_triangles_with_missing_values():
    figure, axis = viz.plt.subplots()
    try:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        tri = viz.mtri.Triangulation(
            points[:, 0], points[:, 1], np.array([[0, 1, 2], [1, 3, 2]])
        )
        contours = viz.draw_scalar_isolines(
            axis, tri, np.array([0.0, 1.0, 2.0, np.nan]), 0.0, 2.0
        )
        assert contours is not None
    finally:
        viz.plt.close(figure)


def test_draw_triangulation_boundary_excludes_internal_edges():
    figure, axis = viz.plt.subplots()
    try:
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        tri = viz.mtri.Triangulation(
            points[:, 0], points[:, 1], np.array([[0, 1, 2], [1, 3, 2]])
        )
        artists = viz.draw_triangulation_boundary(axis, tri)
        assert len(artists) == 4
        assert all(artist.get_color() == "dimgray" for artist in artists)
    finally:
        viz.plt.close(figure)


def test_profile_detection_and_mdpa_submodel_part_parsing(tmp_path):
    all_nodes = {
        1: (0.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 3.0, 0.0),
        4: (2.0, 0.0, 0.0),
    }
    profile_nodes = viz.detect_profile_nodes(
        all_nodes,
        [("Profile", {1, 2, 3}, [])],
        profile_x=0.1,
        x_tol=0.2,
    )
    assert profile_nodes == {1, 2, 3}

    automatic_profile_nodes = viz.detect_profile_nodes(
        all_nodes,
        [("Profile", {1, 2, 3}, [])],
        profile_x=None,
        x_tol=0.2,
    )
    assert automatic_profile_nodes == {1, 2, 3}

    sliced_profile_nodes = viz.detect_profile_nodes(
        all_nodes,
        [],
        profile_x=2.0,
        x_tol=0.2,
    )
    assert sliced_profile_nodes == {4}

    mdpa_path = tmp_path / "model.mdpa"
    mdpa_path.write_text(
        "Begin SubModelPart active\n"
        "Begin SubModelPartNodes\n"
        "1\n2\n"
        "End SubModelPartNodes\n"
        "Begin SubModelPartElements\n"
        "10\n"
        "End SubModelPartElements\n"
        "Begin SubModelPartConditions\n"
        "20\n"
        "End SubModelPartConditions\n"
        "End SubModelPart\n",
        encoding="utf-8",
    )

    active = viz.parse_submodelpart_entities(mdpa_path, "active")
    assert active == viz.ActiveSubmodelPart("active", {1, 2}, {10, 20})


def test_export_line_csv_runs_end_to_end(tmp_path):
    mesh_path = tmp_path / "model.post.msh"
    result_path = tmp_path / "model.post.res"
    write_mesh(mesh_path)
    write_results(result_path)
    config_path = tmp_path / "extract.json"
    config_path.write_text(
        '{"files": ["model"], "result": "PRESSURE", "step": "latest", '
        '"line": {"start": [0, 0], "end": [0, 1], "samples": 2}}',
        encoding="utf-8",
    )

    output_path = viz.export_line_csv_from_json(config_path)

    with output_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert [row["location"] for row in rows] == ["OnNodes"] * 2
    assert [float(row["value"]) for row in rows] == [10.0, 12.0]
    assert [float(row["gradient"]) for row in rows] == [2.0, 2.0]


def test_comparison_loads_different_meshes_and_saves_shared_plot(tmp_path):
    first_mesh = tmp_path / "first.post.msh"
    first_results = tmp_path / "first.post.res"
    second_mesh = tmp_path / "second.post.msh"
    second_results = tmp_path / "second.post.res"
    write_mesh(first_mesh)
    write_results(first_results)
    write_second_mesh(second_mesh)
    write_second_results(second_results)

    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)
    common, first_only, second_only = viz.compare_result_names(first.headers, second.headers)
    assert common == ["DISPLACEMENT", "PRESSURE"]
    assert first_only == ["STRESS"]
    assert second_only == ["TEMPERATURE"]

    output_path = tmp_path / "comparison.png"
    viz.render_comparison(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
        out_path=output_path,
    )

    assert output_path.is_file()

    difference_path = tmp_path / "difference.png"
    viz.render_comparison(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="difference",
        vmin=None,
        vmax=None,
        out_path=difference_path,
    )
    assert difference_path.is_file()


def test_load_solution_field_handles_locations_unknown_nodes_and_reductions(tmp_path, monkeypatch):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    solution = viz.load_solution_data("first", "first", base_dir=tmp_path)

    fallback_field = viz.load_solution_field(
        solution, "PRESSURE", "latest", "OnGaussPoints", "magnitude", "trace"
    )
    assert fallback_field["location"] == "OnNodes"
    assert fallback_field["scalar"][0] == 10.0

    gauss_field = viz.load_solution_field(
        solution, "STRESS", "latest", None, "magnitude", "trace"
    )
    assert gauss_field["location"] == "OnGaussPoints"
    assert np.all(np.isfinite(gauss_field["scalar"]))

    vector_field = viz.load_solution_field(
        solution, "DISPLACEMENT", "latest", None, "y", "trace"
    )
    assert vector_field["result_type"] == "Vector"
    assert vector_field["scalar"][0] == 4.0

    original_load_values = viz.load_values
    monkeypatch.setattr(
        viz,
        "load_values",
        lambda *args: ("Scalar", {999: np.array([42.0])}),
    )
    try:
        unknown_field = viz.load_solution_field(
            solution, "PRESSURE", "latest", "OnNodes", "magnitude", "trace"
        )
    finally:
        monkeypatch.setattr(viz, "load_values", original_load_values)
    assert np.all(np.isnan(unknown_field["scalar"]))


def test_comparison_viewer_update_plot_supports_all_styles(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)

    viewers = []
    try:
        for style in viz.ComparisonViewer.styles:
            viewer = viz.ComparisonViewer(
                first,
                second,
                initial_result="PRESSURE",
                initial_step="latest",
                initial_location=None,
                component="magnitude",
                tensor_component_mode="trace",
                profile_x=0.0,
                style=style,
                vmin=None,
                vmax=None,
            )
            viewer.update_plot()
            assert viewer.current_style == style
            assert viewer.fig is not None
            viewers.append(viewer)
    finally:
        for viewer in viewers:
            close_viewer_figures(viewer)


def test_comparison_viewer_shows_comparison_error_on_all_axes(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    first = viz.load_solution_data("first", "first", base_dir=tmp_path)
    second = viz.load_solution_data("second", "second", base_dir=tmp_path)

    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
        show_profile=True,
    )
    try:
        viewer._show_comparison_error(RuntimeError("comparison failed"))

        assert any(
            text.get_text() == "Error: comparison failed"
            for text in viewer.ax0.texts
        )
        assert not viewer.ax0.axison
        assert not viewer.ax1.axison
        assert not viewer.ax2.axison
    finally:
        close_viewer_figures(viewer)


def load_comparison_solutions(tmp_path):
    write_mesh(tmp_path / "first.post.msh")
    write_results(tmp_path / "first.post.res")
    write_second_mesh(tmp_path / "second.post.msh")
    write_second_results(tmp_path / "second.post.res")
    return (
        viz.load_solution_data("first", "first", base_dir=tmp_path),
        viz.load_solution_data("second", "second", base_dir=tmp_path),
    )


def test_comparison_popup_table_reports_both_nodes_and_missing_variables(tmp_path):
    first, second = load_comparison_solutions(tmp_path)

    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
    )
    try:
        assert viewer.ax1 is None
        columns, rows = viewer._build_comparison_table(0.0, 0.0)
        viewer.inspection_method = "idw"
        idw_columns, idw_rows = viewer._build_comparison_table(0.5, 0.5)
        assert "IDW at (0.5, 0.5)" in idw_columns[1]
        assert "nearest node 1" in idw_columns[1]
        assert "IDW at (0.5, 0.5)" in idw_columns[2]
        assert "nearest node 101" in idw_columns[2]
        assert len(idw_rows) == len(rows)
        viewer._show_comparison_popup(0.5, 0.5)
        assert viewer.info_fig is not None
        viewer._show_comparison_popup(0.75, 0.75)
    finally:
        close_viewer_figures(viewer)

    assert "node 1" in columns[1]
    assert "node 101" in columns[2]
    rows_by_name = {row[0]: row for row in rows}
    assert "10" in rows_by_name["PRESSURE"][1]
    assert "20" in rows_by_name["PRESSURE"][2]
    assert rows_by_name["STRESS"][2] == "-"
    assert rows_by_name["TEMPERATURE"][1] == "-"


def test_comparison_click_and_coordinate_input_open_popup(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
    )
    try:
        popup_coordinates = []
        viewer._show_comparison_popup = lambda x, y: popup_coordinates.append((x, y))
        viewer.on_click(SimpleNamespace(
            inaxes=viewer.ax0, button=1, xdata=0.25, ydata=0.5
        ))
        viewer.on_click(SimpleNamespace(
            inaxes=viewer.ax0, button=3, xdata=0.25, ydata=0.5
        ))
        viewer.on_click(SimpleNamespace(
            inaxes=None, button=1, xdata=0.25, ydata=0.5
        ))
        viewer.on_click(SimpleNamespace(
            inaxes=viewer.ax0, button=1, xdata=None, ydata=0.5
        ))
        assert popup_coordinates == [(0.25, 0.5)]

        viewer._open_coordinate_input_popup()
        viewer.coord_textbox._observers.process("submit", "invalid")
        assert "exactly two numbers" in viewer.coord_status_artist.get_text()
        viewer.coord_textbox._observers.process("submit", "x, y")
        assert "Invalid numeric format" in viewer.coord_status_artist.get_text()
        viewer.coord_textbox._observers.process("submit", "0.75, 1.25")
        assert popup_coordinates == [(0.25, 0.5), (0.75, 1.25)]
    finally:
        close_viewer_figures(viewer)


def test_comparison_mesh_toggle_updates_lines(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = viz.ComparisonViewer(
        first,
        second,
        initial_result="PRESSURE",
        initial_step="latest",
        initial_location=None,
        component="magnitude",
        tensor_component_mode="trace",
        profile_x=0.0,
        style="background+isolines",
        vmin=None,
        vmax=None,
    )
    try:
        viewer.show_mesh = True
        viewer.update_plot()
        mesh_line_styles = {line.get_linestyle() for line in viewer.ax0.lines}
        assert "-" in mesh_line_styles
        assert "--" in mesh_line_styles
    finally:
        close_viewer_figures(viewer)


def make_comparison_viewer(first, second, **overrides):
    options = {
        "initial_result": "PRESSURE",
        "initial_step": "latest",
        "initial_location": None,
        "component": "magnitude",
        "tensor_component_mode": "trace",
        "profile_x": 0.0,
        "style": "background+isolines",
        "vmin": None,
        "vmax": None,
    }
    options.update(overrides)
    return viz.ComparisonViewer(first, second, **options)


def test_comparison_profile_and_menu_dragging(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        colormap="cividis",
    )
    try:
        assert viewer.ax1 is not None
        assert viewer.ax2 is not None
        assert viewer.current_colormap == "cividis"
        profile_line_count = len(viewer.ax1.lines)
        profile_data = viewer._draw_profile(
            viewer.solutions[0], viewer.fields[0], "green"
        )
        assert profile_data is not None
        assert len(viewer.ax1.lines) == profile_line_count + 1

        menu_artist = viewer.menu_artists[0]
        viewer._on_menu_pick(SimpleNamespace(
            artist=menu_artist,
            mouseevent=SimpleNamespace(x=100, y=100),
        ))
        viewer._on_menu_motion(SimpleNamespace(x=150, y=140))
        viewer._on_menu_release(SimpleNamespace())
        assert viewer.menu_drag_artist is None
    finally:
        close_viewer_figures(viewer)


def test_comparison_viewer_renders_profile_path(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        profile_path=[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
    )
    try:
        assert viewer.ax1 is not None
        assert viewer.ax1.get_ylabel() == "distance"
        assert viewer._get_profile_data(viewer.solutions[0], viewer.fields[0]) is not None
    finally:
        close_viewer_figures(viewer)


def test_comparison_settings_and_scale_input(tmp_path, monkeypatch):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        colormap="cividis",
    )
    try:
        picture_path = viewer.save_current_plot(tmp_path / "hotkey_picture.png")
        assert picture_path.is_file()
        settings_path = viewer.save_current_settings(tmp_path / "comparison_settings.json")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings["mode"] == "comparison"
        assert len(settings["solutions"]) == 2

        viewer._open_scale_input_popup()
        viewer.scale_textbox._observers.process("submit", "invalid")
        assert "exactly two numbers" in viewer.scale_status_artist.get_text()
        viewer.scale_textbox._observers.process("submit", "10, 5")
        assert "vmin < vmax" in viewer.scale_status_artist.get_text()
        viewer.scale_textbox._observers.process("submit", "-10, 50")
        assert (viewer.vmin, viewer.vmax) == (-10.0, 50.0)
        monkeypatch.chdir(tmp_path)
        assert viewer.save_current_plot().is_file()
    finally:
        close_viewer_figures(viewer)


def test_comparison_keyboard_dispatch(tmp_path):
    first, second = load_comparison_solutions(tmp_path)
    viewer = make_comparison_viewer(
        first,
        second,
        show_profile=True,
        colormap="cividis",
    )
    try:
        calls = []
        viewer.update_plot = lambda: calls.append("update")
        viewer.save_current_plot = lambda: calls.append("save")
        viewer._open_coordinate_input_popup = lambda: calls.append("coordinate")
        viewer._open_scale_input_popup = lambda: calls.append("scale")
        for key in ("n", "p", "o", "c", "l", "L", "m", "b", "a", "w", "j", "k", "[", "]", "i", "S", "v"):
            viewer.on_key(SimpleNamespace(key=key))
        viewer.on_key(SimpleNamespace(key="s"))
        assert "save" in calls
        assert "coordinate" in calls
        assert "scale" in calls
        assert calls.count("update") == 14
        viewer.on_key(SimpleNamespace(key="q"))
    finally:
        close_viewer_figures(viewer)

def test_comparison_viewer_rejects_invalid_options(tmp_path):
    first, second = load_comparison_solutions(tmp_path)

    with pytest.raises(ValueError, match="Inspection method"):
        make_comparison_viewer(
            first,
            second,
            inspection_method="invalid",
        )

    with pytest.raises(ValueError, match="Unsupported comparison colormap"):
        make_comparison_viewer(
            first,
            second,
            colormap="invalid",
        )