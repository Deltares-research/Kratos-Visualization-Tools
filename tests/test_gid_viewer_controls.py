from types import SimpleNamespace

import numpy as np

import visualize_gid_results as viz

from conftest import (
    close_viewer_figures,
    make_single_viewer,
    write_mesh,
    write_results,
    write_second_mesh,
    write_second_results,
)


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
