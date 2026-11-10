import json
import sys
import xml.etree.ElementTree as ET

import pytest

from kratos_visualization_tools.svg import cli


def test_main_converts_msh_using_defaults(tmp_path, monkeypatch, capsys):
    source = tmp_path / "model.msh"
    source.write_text(
        "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n"
        "$Elements\n1\n1 2 0 1 2 3\n$EndElements\n",
        encoding="utf-8",
    )
    output = tmp_path / "mesh.svg"
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", str(source), "--output", str(output),
    ])

    cli.main()

    assert ET.parse(output).getroot().attrib["viewBox"] == "0 0 2200 1200"
    assert "Parsed nodes=3, elements=1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config_value", "message"),
    [
        ([], "Legend config root must be a JSON object"),
        ({"legend": []}, "Legend config must contain a JSON object"),
    ],
)
def test_main_rejects_non_object_legend_config(tmp_path, monkeypatch, config_value, message):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(config_value), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", "unused.geo", "--output", str(tmp_path / "out.svg"),
        "--legend-config", str(config),
    ])

    with pytest.raises(ValueError, match=message):
        cli.main()


def test_main_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", "model.txt", "--output", str(tmp_path / "out.svg"),
    ])

    with pytest.raises(ValueError, match="Unsupported input extension"):
        cli.main()


def test_main_converts_mdpa_with_existing_legend_config(
    mdpa_svg_input, tmp_path, monkeypatch, capsys
):
    source = mdpa_svg_input.with_suffix(".MDPA")
    mdpa_svg_input.rename(source)
    output = tmp_path / "nested" / "mdpa.svg"
    config = tmp_path / "legend.json"
    config.write_text(json.dumps({"legend": {
        "surfaces": [{"name": "Soil", "label": "Ground"}],
    }}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", str(source), "--output", str(output),
        "--width", "400", "--height", "300", "--padding", "20",
        "--legend-config", str(config),
    ])

    cli.main()

    root = ET.parse(output).getroot()
    assert root.attrib["viewBox"] == "0 0 400 300"
    assert any(node.text == "Ground" for node in root.iter())
    stdout = capsys.readouterr().out
    assert f"Wrote SVG: {output}" in stdout
    # Bottom (condition) plus NodesOnly (node-only SubModelPart on an outer edge).
    assert "Physical curves=2, physical surfaces=1" in stdout
