import pytest

import visualize_gid_results as viz

from conftest import (
    write_mesh,
    write_results,
    write_second_mesh,
    write_second_results,
)


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
