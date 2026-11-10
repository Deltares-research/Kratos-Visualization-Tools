import xml.etree.ElementTree as ET

import pytest

from kratos_visualization_tools.svg import document, mdpa_reader, models, render
from conftest import _square_model, _surface_paths, _two_square_model


SVG = f"{{{document.SVG_NS}}}"


@pytest.mark.parametrize(
    ("renderer", "model", "message"),
    [
        ("geo_to_svg", models.GeoModel(), "No points parsed"),
        ("msh_to_svg", models.MshModel(), "No nodes parsed"),
        ("msh_to_svg", models.MshModel(nodes={1: (0, 0, 0)}), "No 2D mesh elements"),
    ],
)
def test_renderers_report_missing_geometry(tmp_path, renderer, model, message):
    with pytest.raises(ValueError, match=message):
        getattr(render, renderer)(model, tmp_path / "invalid.svg", 100, 100, 5)


def test_bottom_hatching_is_evenly_spaced_for_unordered_edges(tmp_path):
    model = models.GeoModel(
        points={i: (float(i), 0.0) for i in range(11)} | {11: (0.0, 5.0)},
        splines={i + 1: [n, n + 1] for i, n in enumerate([7, 2, 9, 0, 4, 1, 8, 3, 6, 5])},
        physical_curves={"Bottom": list(range(1, 11))},
    )
    output = tmp_path / "hatch.svg"

    render.geo_to_svg(model, output, 520, 300, 10, legend_cfg={"rulers": False})

    lines = ET.parse(output).getroot().findall(f"{SVG}g[@id='supports_and_loads']/{SVG}line")
    xs = sorted(float(line.attrib["x1"]) for line in lines)
    assert len(xs) == 22
    assert all(b - a == pytest.approx(22.0) for a, b in zip(xs, xs[1:]))


def test_geo_to_svg_writes_geometry_and_configured_legend(tmp_path):
    model = models.GeoModel(
        points={1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)},
        lines={1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 1)},
        curve_loops={1: [1, 2, 3, 4]},
        plane_surfaces={1: [1]},
        physical_curves={"Boundary": [1]},
        physical_surfaces={"Soil": [1]},
    )
    output = tmp_path / "nested" / "preview.svg"

    render.geo_to_svg(
        model,
        output,
        width=400,
        height=300,
        padding=20,
        legend_cfg={
            "curves": [{"name": "Boundary", "label": "Model boundary"}],
            "phreatic_line": {"label": "Water level", "points": [[0, 0.5], [1, 0.5]]},
            "top_left": [12, 14],
        },
    )

    root = ET.parse(output).getroot()
    assert root.attrib["viewBox"] == "0 0 400 300"
    surface_paths = root.findall(f".//{SVG}g[@id='physical_surfaces']/{SVG}path")
    fills = [path for path in surface_paths if path.attrib["stroke"] == "none"]
    assert len(fills) == 2
    assert all(path.attrib["fill-rule"] == "evenodd" for path in fills)
    assert not root.findall(f".//{SVG}g[@id='physical_surfaces']/{SVG}polygon")
    assert len(root.findall(f".//{SVG}g[@id='physical_curves']/{SVG}path")) == 1
    assert root.find(f".//{SVG}g[@id='legend']/{SVG}text").text == "Physical Curves (conditions)"
    assert any(node.text == "Model boundary" for node in root.findall(f".//{SVG}g[@id='legend']/{SVG}text"))
    assert any(node.text == "Water level" for node in root.findall(f".//{SVG}g[@id='legend']/{SVG}text"))
    assert root.find(f".//{SVG}g[@id='supports_and_loads']/{SVG}path") is not None


def test_msh_to_svg_draws_boundary_conditions_and_shared_edges(tmp_path):
    model = models.MshModel(
        nodes={
            1: (0, 0, 0),
            2: (1, 0, 0),
            3: (0, 1, 0),
            4: (1, 1, 0),
        },
        physical_names={1: (2, "Left"), 2: (2, "Right"), 3: (1, "Boundary")},
        elements=[
            (1, 2, [1, 2, 3], [1]),
            (2, 2, [2, 4, 3], [2]),
        ],
        line_elements=[(3, 1, [1, 2], [3])],
    )
    output = tmp_path / "mesh.svg"

    render.msh_to_svg(model, output, width=400, height=300, padding=20)

    root = ET.parse(output).getroot()
    elements = root.find(f".//{SVG}g[@id='mesh_elements']")
    shared = root.find(f".//{SVG}g[@id='mesh_shared_edges']")
    assert elements is not None
    assert shared is not None
    assert len(elements.findall(f"{SVG}path")) >= 1
    assert any(path.attrib["stroke"] == document._hash_color("Boundary") for path in elements.findall(f"{SVG}path"))
    assert any(path.attrib["stroke"] == "#000000" for path in shared.findall(f"{SVG}path"))


def test_geo_to_svg_skips_missing_surfaces_short_loops_and_curves(tmp_path):
    model = models.GeoModel(
        points={1: (0, 0), 2: (1, 1)},
        lines={1: (1, 2)},
        curve_loops={1: [1]},
        plane_surfaces={1: [1]},
        physical_surfaces={"Invalid": [99, 1]},
        physical_curves={"Invalid": [99]},
    )
    output = tmp_path / "partial.svg"

    render.geo_to_svg(model, output, 100, 100, 5)

    root = ET.parse(output).getroot()
    assert len(root.find(f"{SVG}g[@id='physical_surfaces']")) == 0
    assert len(root.find(f"{SVG}g[@id='physical_curves']")) == 0


@pytest.mark.parametrize("same_surface", [False, True])
def test_msh_to_svg_colors_shared_edges_by_surface_when_enabled(
    tmp_path, two_triangle_mesh, same_surface
):
    if same_surface:
        two_triangle_mesh.elements[1] = (2, 2, [2, 4, 3], [1])
    output = tmp_path / "shared.svg"

    render.msh_to_svg(
        two_triangle_mesh, output, 100, 100, 5,
        legend_cfg={"msh": {"shared_edges_black": False, "boundary_edge_width": 2}},
    )

    root = ET.parse(output).getroot()
    paths = root.findall(f"{SVG}g[@id='mesh_shared_edges']/{SVG}path")
    assert len(paths) == 1
    allowed_colors = {document._hash_color("Clay")}
    if not same_surface:
        allowed_colors.add(document._hash_color("Sand"))
    assert paths[0].attrib["stroke"] in allowed_colors
    assert paths[0].attrib["stroke-width"] == "2.00"


def test_msh_to_svg_skips_invalid_elements_and_uses_unnamed_element_style(
    tmp_path, two_triangle_mesh
):
    two_triangle_mesh.elements = [
        (1, 2, [1, 2, 3], [999]),
        (2, 2, [1, 2], []),
        (3, 2, [1, 2, 99], []),
        (4, 4, [1, 2, 3, 4], []),
    ]
    two_triangle_mesh.line_elements = [
        (5, 1, [1], []),
        (6, 2, [1, 2, 3], []),
        (7, 1, [1, 2], [1, 999]),
    ]
    output = tmp_path / "fallback.svg"

    render.msh_to_svg(
        two_triangle_mesh, output, 100, 100, 5, legend_cfg={"msh": []}
    )

    root = ET.parse(output).getroot()
    paths = root.findall(f"{SVG}g[@id='mesh_elements']/{SVG}path")
    assert len(paths) == 1
    assert paths[0].attrib["stroke"] == document._hash_color("elem_1")
    assert any(node.text == "Elements rendered: 1 (line-only view)" for node in root.iter())


def test_mdpa_rendering_preserves_region_borders_legend_and_symbols(
    mdpa_svg_input, tmp_path
):
    output = tmp_path / "mdpa.svg"
    model = mdpa_reader.parse_mdpa(mdpa_svg_input)

    render.geo_to_svg(
        model, output, 400, 300, 20,
        legend_cfg={
            "surfaces": [{"name": "Soil", "label": "Ground"}],
            "curves": [{"name": "Bottom", "label": "Fixed base"}],
            "phreatic_line": {"points": [[0, 0.5], [1, 0.5]]},
        },
    )

    root = ET.parse(output).getroot()
    surfaces = root.find(f"{SVG}g[@id='physical_surfaces']")
    fills = [path for path in surfaces.findall(f"{SVG}path") if path.attrib["stroke"] == "none"]
    assert len(fills) == 2
    assert all(path.attrib["d"].count("Z") == 1 for path in fills)
    borders = [path for path in surfaces.findall(f"{SVG}path") if path.attrib["fill"] == "none"]
    assert len(borders) == 1
    assert borders[0].attrib["d"].count("M") == 1
    assert borders[0].attrib["d"].count("L") == 3
    assert fills[0].attrib["d"] == borders[0].attrib["d"]
    assert len(root.findall(f"{SVG}g[@id='physical_curves']/{SVG}path")) == 2
    assert root.findall(f"{SVG}g[@id='supports_and_loads']/{SVG}line")
    assert root.find(f"{SVG}g[@id='supports_and_loads']/{SVG}path") is not None
    labels = [node.text for node in root.findall(f"{SVG}g[@id='legend']/{SVG}text")]
    assert "Ground" in labels
    assert "Fixed base" in labels


@pytest.mark.parametrize(
    ("surfaces_cfg", "fill_count", "corner_counts"),
    [
        ([{"names": ["Left", "Right"], "label": "Stages"}], 2, [4]),
        (None, 4, [4, 4]),
    ],
)
def test_geo_to_svg_merges_grouped_regions_into_one_outline(
    tmp_path, surfaces_cfg, fill_count, corner_counts
):
    output = tmp_path / "merged.svg"

    render.geo_to_svg(
        _two_square_model(Left=[1], Right=[2]), output, 400, 300, 20,
        legend_cfg={"surfaces": surfaces_cfg},
    )

    fills, borders = _surface_paths(output)
    assert len(fills) == fill_count
    assert all(path.attrib["d"].count("M") == 1 for path in borders)
    assert sorted(path.attrib["d"].count("L") + 1 for path in borders) == corner_counts


def test_geo_to_svg_fills_from_outline_and_keeps_holes(tmp_path):
    output = tmp_path / "hole.svg"
    model = models.GeoModel(
        points={
            1: (0, 0), 2: (4, 0), 3: (4, 4), 4: (0, 4),
            5: (1, 1), 6: (3, 1), 7: (3, 3), 8: (1, 3),
        },
        splines={1: [1, 2, 3, 4, 1], 2: [5, 6, 7, 8, 5]},
        curve_loops={1: [1], 2: [2]},
        plane_surfaces={1: [1, 2]},
        physical_surfaces={"Ring": [1]},
    )

    render.geo_to_svg(model, output, 400, 400, 20)

    fills, borders = _surface_paths(output)
    assert {path.attrib["d"].count("Z") for path in fills} == {2}
    assert all(path.attrib["fill-rule"] == "evenodd" for path in fills)
    assert borders[0].attrib["d"].count("M") == 2


def test_geo_to_svg_draws_shared_surfaces_and_curves_once(tmp_path):
    output = tmp_path / "dedup.svg"
    model = _two_square_model(aggregate=[1, 2], Left=[1], Right=[2])
    model.lines = {7: (1, 2)}
    model.physical_curves = {"Bottom": [7], "all_curves": [7, -7]}

    render.geo_to_svg(model, output, 400, 300, 20)

    fills, borders = _surface_paths(output)
    assert len(fills) == 4
    assert all(path.attrib["d"].count("Z") == 1 for path in fills)
    assert len(borders) == 2
    root = ET.parse(output).getroot()
    assert len(root.findall(f"{SVG}g[@id='physical_curves']/{SVG}path")) == 1


def test_geo_to_svg_draws_coincident_interface_faces_once(tmp_path):
    output = tmp_path / "interface.svg"
    model = models.GeoModel(
        points={1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 0), 5: (1, 0), 6: (2, 0)},
        splines={1: [1, 2], 2: [2, 3], 3: [4, 5], 4: [5, 6]},
        physical_curves={"Interface": [1, 2, 3, 4]},
    )

    render.geo_to_svg(model, output, 400, 300, 20)

    paths = ET.parse(output).getroot().findall(f"{SVG}g[@id='physical_curves']/{SVG}path")
    assert len(paths) == 1
    assert paths[0].attrib["d"].count("M") == 1
    assert paths[0].attrib["d"].count("L") == 1


def test_geo_to_svg_uses_configured_colors_in_drawing_and_legend(tmp_path):
    output = tmp_path / "colors.svg"

    render.geo_to_svg(_square_model(), output, 400, 300, 20, legend_cfg={
        "elements": [{"name": "Clay", "color": "#ABCDEF"}],
        "conditions": [{"name": "Top", "color": "#123456"}],
    })

    root = ET.parse(output).getroot()
    surface_paths = root.findall(f"{SVG}g[@id='physical_surfaces']/{SVG}path")
    assert any(path.attrib["fill"] == "#abcdef" for path in surface_paths)
    assert any(path.attrib["stroke"] == "#abcdef" for path in surface_paths)
    curve_paths = root.findall(f"{SVG}g[@id='physical_curves']/{SVG}path")
    assert [path.attrib["stroke"] for path in curve_paths] == ["#123456"]
    legend = root.find(f"{SVG}g[@id='legend']")
    assert legend.find(f"{SVG}line").attrib["stroke"] == "#123456"
    assert legend.find(f"{SVG}rect").attrib["stroke"] == "#abcdef"


def test_geo_to_svg_draws_rulers_and_axis_triad_by_default(tmp_path):
    output = tmp_path / "rulers.svg"

    render.geo_to_svg(_square_model(), output, 400, 300, 20)

    root = ET.parse(output).getroot()
    rulers = root.find(f"{SVG}g[@id='coordinate_rulers']")
    assert [node.text for node in rulers.findall(f"{SVG}text")] == ["0", "5", "10", "0", "5"]
    assert len(rulers.findall(f"{SVG}line")) == 5
    assert rulers.find(f"{SVG}path").attrib["d"] == "M20.0,20.0V280.0H380.0"
    triad = root.find(f"{SVG}g[@id='axes_triad']")
    assert [node.text for node in triad.findall(f"{SVG}text")] == ["X", "Y"]
    assert len(triad.findall(f"{SVG}polygon")) == 2


def test_geo_to_svg_hides_rulers_when_disabled(tmp_path):
    output = tmp_path / "no_rulers.svg"

    render.geo_to_svg(_square_model(), output, 400, 300, 20, legend_cfg={"rulers": False})

    root = ET.parse(output).getroot()
    assert root.find(f"{SVG}g[@id='coordinate_rulers']") is None
    assert root.find(f"{SVG}g[@id='axes_triad']") is None


@pytest.mark.parametrize("rulers", [True, False])
def test_msh_to_svg_uses_configured_colors_and_optional_rulers(tmp_path, rulers):
    model = models.MshModel(
        nodes={1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0)},
        physical_names={1: (2, "Soil"), 2: (1, "Base")},
        elements=[(1, 2, [1, 2, 3], [1])],
        line_elements=[(2, 1, [1, 2], [2])],
    )
    output = tmp_path / "mesh_colors.svg"

    render.msh_to_svg(model, output, 400, 300, 20, legend_cfg={
        "elements": [{"name": "Soil", "color": "#00FF00"}],
        "conditions": [{"name": "Base", "color": "#0000FF"}],
        "rulers": rulers,
    })

    root = ET.parse(output).getroot()
    strokes = {path.attrib["stroke"] for path in root.findall(f"{SVG}g[@id='mesh_elements']/{SVG}path")}
    assert strokes == {"#00ff00", "#0000ff"}
    assert (root.find(f"{SVG}g[@id='coordinate_rulers']") is not None) is rulers
    assert (root.find(f"{SVG}g[@id='axes_triad']") is not None) is rulers
