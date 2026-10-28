import json

import pytest

import visualize_gid_results as viz


@pytest.fixture
def case_dir(tmp_path, mesh_file):
    write_stage_output(tmp_path, "stage_1", 0.0, {1: 1000.0, 2: 2000.0, 3: 3000.0})
    write_stage_output(tmp_path, "stage_2", 1.0, {1: 0.0, 2: 4000.0, 3: 6000.0})
    return tmp_path


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


@pytest.fixture
def mesh_file(tmp_path):
    path = tmp_path / "model.mdpa"
    path.write_text(
        "Begin ModelPartData\nEnd ModelPartData\n"
        "Begin Nodes\n"
        "  1 0.0  0.0 0.0\n"
        "  2 1.0 -1.0 0.0\n"
        "  3 2.0 -2.0 0.0\n"
        "End Nodes\n"
        "Begin Elements UPwSmallStrainElement2D3N\n"
        "  1 0 1 2 3\n"
        "End Elements\n"
        "Begin SubModelPart wall\n"
        "  Begin SubModelPartNodes\n"
        "    1\n"
        "    2\n"
        "    3\n"
        "  End SubModelPartNodes\n"
        "  Begin SubModelPart wall_tip\n"
        "    Begin SubModelPartNodes\n"
        "      3\n"
        "    End SubModelPartNodes\n"
        "  End SubModelPart\n"
        "End SubModelPart\n"
    )
    return path


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


def write_stage_output(directory, stage, time, values_by_node_id):
    data = {"TIME": [time]}
    for node_id, value in values_by_node_id.items():
        data[f"NODE_{node_id}"] = {
            "BENDING_MOMENT": [value],
            "TRACTION": [[value, 2.0 * value]],
        }
    (directory / f"{stage}__output.json").write_text(json.dumps(data))
