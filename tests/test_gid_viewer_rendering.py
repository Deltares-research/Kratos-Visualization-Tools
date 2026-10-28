from types import SimpleNamespace

import numpy as np
import pytest

import visualize_gid_results as viz

from conftest import (
    close_viewer_figures,
    make_single_viewer,
    write_mesh,
    write_results,
)


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
