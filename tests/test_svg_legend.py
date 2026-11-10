import pytest

from kratos_visualization_tools.svg import document, legend, models


@pytest.mark.parametrize("kind", ["curve", "surface"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Soil ", (["Soil"], "Soil")),
        ("  ", None),
        (None, None),
        (42, None),
        (["Soil"], None),
        ({}, None),
        ({"name": "Soil", "show": False}, None),
        ({"name": "  "}, None),
        ({"name": 42}, None),
        ({"name": " Soil "}, (["Soil"], "Soil")),
        ({"name": "Soil", "label": " Ground "}, (["Soil"], "Ground")),
        ({"name": "Soil", "label": ""}, (["Soil"], "Soil")),
        ({"name": "Soil", "label": " "}, (["Soil"], "Soil")),
        ({"name": "Soil", "label": 7}, (["Soil"], "7")),
        ({"names": []}, None),
        ({"names": [" ", ""]}, None),
        (
            {"names": [" Clay ", "", " Sand "], "label": " Ground "},
            (["Clay", "Sand"], "Ground"),
        ),
        ({"names": ["Clay", "Sand"]}, (["Clay", "Sand"], "Clay, Sand")),
        (
            {"name": "ignored", "names": ["Clay", "Sand"], "label": " "},
            (["Clay", "Sand"], "Clay"),
        ),
    ],
)
def test_parse_legend_entry_normalizes_filters_and_groups_names(raw, expected, kind):
    assert legend._parse_legend_entry(raw, kind) == expected


def test_build_legend_spec_filters_entries_and_groups_surface_styles():
    model = models.GeoModel(
        physical_curves={"Boundary": [1]},
        physical_surfaces={"Clay": [1], "Sand": [2]},
    )

    spec = legend._build_legend_spec(
        model,
        {
            "curves": [None, "missing", "Boundary"],
            "surfaces": [
                {"name": "Clay", "show": False},
                "missing",
                {"names": ["Clay", "missing", "Sand"], "label": "Ground"},
            ],
            "max_curves": 1,
            "max_surfaces": 2,
        },
    )

    assert [entry.names for entry in spec.curve_entries] == [["Boundary"]]
    assert [entry.names for entry in spec.surface_entries] == [["Clay", "Sand"]]
    assert spec.surface_entries[0].label == "Ground"
    assert spec.surface_style_by_name["Clay"] == spec.surface_style_by_name["Sand"]
    assert spec.surface_entries[0].style_key == spec.surface_style_by_name["Clay"]
    assert (spec.max_curve_items, spec.max_surface_items) == (1, 2)


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], "invalid"],
        [[0, 0], ["invalid", 1]],
        [[0, 0], [None, 1]],
    ],
)
def test_build_legend_spec_ignores_invalid_phreatic_points(points):
    spec = legend._build_legend_spec(
        models.GeoModel(), {"phreatic_line": {"points": points}}
    )

    assert spec.phreatic_line_points is None


@pytest.mark.parametrize(
    ("anchor", "expected"),
    [
        (["invalid", 2], (20, 20)),
        ({"x": "12", "y": "34"}, (12, 34)),
        ({"x": 12}, (12, 20)),
        ({"x": None}, (20, 20)),
    ],
)
def test_build_legend_spec_parses_anchor_and_handles_invalid_values(anchor, expected):
    spec = legend._build_legend_spec(models.GeoModel(), {"top_left": anchor})

    assert spec.legend_top_left == expected


@pytest.mark.parametrize("kind", ["curve", "surface"])
def test_legend_entries_keep_group_indices_and_last_style_assignment(kind):
    entries, styles = legend._legend_entries(
        ["Clay", "Sand", "Other"],
        [
            None,
            {"names": ["Clay", "Sand", "missing"], "label": "Ground"},
            {"name": "Clay", "label": "Top"},
        ],
        kind,
    )

    assert entries == [
        legend.LegendEntry(["Clay", "Sand"], "Ground", f"{kind}_cfg_1_Ground"),
        legend.LegendEntry(["Clay"], "Top", f"{kind}_cfg_2_Top"),
    ]
    assert styles == {
        "Clay": f"{kind}_cfg_2_Top",
        "Sand": f"{kind}_cfg_1_Ground",
        "Other": "Other",
    }


@pytest.mark.parametrize("config", [None, {}, {"top_left": [20, 20]}])
def test_build_legend_spec_preserves_sorted_defaults(config):
    model = models.GeoModel(
        physical_curves={"Z": [], "A": []},
        physical_surfaces={"Sand": [], "Clay": []},
    )

    spec = legend._build_legend_spec(model, config)

    assert [entry.label for entry in spec.curve_entries] == ["A", "Z"]
    assert [entry.label for entry in spec.surface_entries] == ["Clay", "Sand"]
    assert spec.curve_style_by_name == {"A": "A", "Z": "Z"}
    assert spec.surface_style_by_name == {"Clay": "Clay", "Sand": "Sand"}
    assert (spec.max_curve_items, spec.max_surface_items) == (18, 18)


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        ([], {"aggregate": [], "left": [1, 2], "right": [3]}),
        (
            [legend.LegendEntry(["aggregate"], "All", "all")],
            {"aggregate": [1, 2, 3], "left": [], "right": []},
        ),
        (
            [legend.LegendEntry(["missing", "right"], "R", "r")],
            {"aggregate": [], "left": [1, 2], "right": [3]},
        ),
    ],
)
def test_unique_group_ids_prefers_configured_then_smallest_groups(entries, expected):
    groups = {"aggregate": [1, 2, 3], "left": [1, 2], "right": [3, -3]}

    assert legend._unique_group_ids(groups, entries) == expected


def test_surface_regions_group_names_by_legend_style():
    styles = {"A": "ab", "B": "ab", "C": "c", "D": "D"}

    assert legend._surface_regions(["A", "B", "C", "D", "E"], styles) == [
        ("ab", ["A", "B"]),
        ("c", ["C"]),
        ("D", ["D"]),
        ("E", ["E"]),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"color": "#ABCdef"}, "#abcdef"),
        ({"name": "A"}, None),
        ("A", None),
    ],
)
def test_configured_legend_color_returns_lowercase_hex_or_none(raw, expected):
    assert legend._configured_legend_color(raw) == expected


@pytest.mark.parametrize("color", ["red", "#12345", "#1234567", 123])
def test_configured_legend_color_rejects_invalid_values(color):
    with pytest.raises(ValueError, match="#RRGGBB"):
        legend._configured_legend_color({"color": color})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, (14.0, 11.0)),
        ({}, (14.0, 11.0)),
        ({"title": 20}, (20.0, 11.0)),
        ({"title": 18, "label": 13.5}, (18.0, 13.5)),
    ],
)
def test_font_sizes_use_defaults_for_missing_values(raw, expected):
    assert legend._font_sizes(raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (12, "must be an object"),
        ({"title": 0}, '"title" must be a positive number'),
        ({"label": -1}, '"label" must be a positive number'),
        ({"label": "11"}, '"label" must be a positive number'),
        ({"title": True}, '"title" must be a positive number'),
    ],
)
def test_font_sizes_reject_invalid_values(raw, message):
    with pytest.raises(ValueError, match=message):
        legend._font_sizes(raw)


def test_build_legend_spec_reads_font_sizes():
    spec = legend._build_legend_spec(
        models.GeoModel(), {"font_sizes": {"title": 22, "label": 16}}
    )

    assert (spec.title_font_size, spec.label_font_size) == (22.0, 16.0)


def test_build_legend_spec_accepts_conditions_and_elements_aliases():
    model = models.GeoModel(
        physical_curves={"Head": [1], "Other": [2]},
        physical_surfaces={"Clay": [1]},
    )

    spec = legend._build_legend_spec(model, {
        "conditions": [{"name": "Head", "label": "Head = 5 m", "color": "#E34066"}],
        "elements": [{"name": "Clay", "label": "Clay layer"}],
    })

    assert spec.curve_entries == [
        legend.LegendEntry(["Head"], "Head = 5 m", "curve_cfg_0_Head = 5 m", "#e34066"),
    ]
    assert [entry.label for entry in spec.surface_entries] == ["Clay layer"]
    assert spec.color("curve_cfg_0_Head = 5 m") == "#e34066"
    assert spec.color("surface_cfg_0_Clay layer") == document._hash_color("surface_cfg_0_Clay layer")
    assert spec.color("Other") == document._hash_color("Other")


def test_build_legend_spec_prefers_curves_and_surfaces_over_aliases():
    model = models.GeoModel(physical_curves={"A": [1]}, physical_surfaces={"S": [1]})

    spec = legend._build_legend_spec(model, {
        "curves": [{"name": "A", "label": "primary"}],
        "conditions": [{"name": "A", "label": "alias"}],
        "surfaces": [{"name": "S", "label": "primary"}],
        "elements": [{"name": "S", "label": "alias"}],
    })

    assert [entry.label for entry in spec.curve_entries] == ["primary"]
    assert [entry.label for entry in spec.surface_entries] == ["primary"]
