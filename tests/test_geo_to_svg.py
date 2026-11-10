import json
import runpy
import sys
import xml.etree.ElementTree as ET

import pytest

import geo_to_svg as converter
from kratos_visualization_tools import svg
from kratos_visualization_tools.svg import document


SVG = f"{{{document.SVG_NS}}}"


def test_script_reexports_package_api():
    for name in svg.__all__:
        assert getattr(converter, name) is getattr(svg, name)
    assert converter.SVG_NS == document.SVG_NS


@pytest.mark.parametrize("nested", [False, True])
def test_script_entry_point_converts_geo_with_legend_config(
    tmp_path, monkeypatch, capsys, nested
):
    source = tmp_path / "model.GEO"
    source.write_text("Point(1) = {0, 0};", encoding="utf-8")
    output = tmp_path / "cli.svg"
    config = tmp_path / "legend.json"
    legend = {"top_left": [12, 34]}
    config.write_text(json.dumps({"legend": legend} if nested else legend), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "geo_to_svg.py", "--input", str(source), "--output", str(output),
        "--width", "300", "--height", "200", "--padding", "10",
        "--legend-config", str(config),
    ])

    runpy.run_path(converter.__file__, run_name="__main__")

    root = ET.parse(output).getroot()
    assert root.attrib["viewBox"] == "0 0 300 200"
    title = root.find(f"{SVG}g[@id='legend']/{SVG}text")
    assert (title.attrib["x"], title.attrib["y"]) == ("12.0", "34.0")
    stdout = capsys.readouterr().out
    assert f"Wrote SVG: {output}" in stdout
    assert "Parsed points=1, lines=0, splines=0, loops=0" in stdout
    assert "Physical curves=0, physical surfaces=0" in stdout
