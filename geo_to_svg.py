#!/usr/bin/env python3
"""Convert a Gmsh .geo/.msh or Kratos .mdpa file into an SVG preview.

Thin wrapper around :mod:`kratos_visualization_tools.svg`; the same command is
installed as ``kratos-geo-to-svg``. See geo_to_svg.md for the legend config schema.

Usage:
    python geo_to_svg.py --input model.geo  --output model.svg
    python geo_to_svg.py --input model.msh  --output model_mesh.svg
    python geo_to_svg.py --input model.mdpa --output model.svg --legend-config cfg.json
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

from kratos_visualization_tools.svg import (  # noqa: E402
    GeoModel,
    MshModel,
    eval_expr,
    geo_to_svg,
    main,
    msh_to_svg,
    parse_geo,
    parse_mdpa,
    parse_msh,
)
from kratos_visualization_tools.svg.document import SVG_NS  # noqa: E402

__all__ = [
    "GeoModel",
    "MshModel",
    "SVG_NS",
    "eval_expr",
    "geo_to_svg",
    "main",
    "msh_to_svg",
    "parse_geo",
    "parse_mdpa",
    "parse_msh",
]

if __name__ == "__main__":
    main()
