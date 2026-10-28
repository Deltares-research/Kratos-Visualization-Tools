import numpy as np

import visualize_gid_results as viz

from conftest import (
    write_mesh,
    write_results,
)


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
