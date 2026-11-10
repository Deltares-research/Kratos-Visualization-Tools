"""Convert a Gmsh .geo/.msh or Kratos .mdpa file into an SVG preview.

Features:
- Draws physical curves (conditions) with distinct colors.
- Draws physical surfaces (elements) with hatch/dot patterns.
- If input is .msh, draws each mesh element with a distinct fill color.
- Supports common GEO entities used in 2D profiles: Point, Line, Spline,
  Curve Loop / Line Loop, Plane Surface, Physical Curve/Line/Surface.
- MDPA input reuses the package parser and draws SubModelParts as regions/curves;
  nested SubModelParts are named "Parent/Child", and node-only SubModelParts
  are drawn along the outer element edges whose end nodes they contain.
- Draws model-coordinate rulers and an X/Y axis triad.

Modules:
- models: GeoModel, MshModel and geometry type aliases
- geo_reader, msh_reader, mdpa_reader: input readers
- geometry: polylines, loops, segment chaining, coordinate transform
- document: SVG skeleton, fill patterns and colours
- legend: legend configuration, styles and group ownership
- symbols: supports, loads, phreatic line, rulers and axis triad
- render: geo_to_svg / msh_to_svg
- cli: command-line entry point (``kratos-geo-to-svg`` / ``geo_to_svg.py``)

Legend config schema (root object, or nested under "legend"):
    - "curves" (alias "conditions"): [{"name": "group", "label": "Legend name", "show": true,
      "color": "#RRGGBB"}, ...]
    - "surfaces" (alias "elements"): [{"name": "group"} or {"names": ["g1", "g2"], "label": "Combined"}, ...]
      (grouped names are drawn as one region with one outline; "color" is optional)
    - "max_curves": int, "max_surfaces": int
    - "rulers": false (hide coordinate rulers and the axis triad)
    - "top_left": [x, y] or {"x": x, "y": y} (legend top-left anchor in SVG pixels)
    - "phreatic_line": {"label": "...", "points": [[x1, y1], [x2, y2], ...]}
    - "symbols": {
          "bottom_hatch_curves": ["Bottom"],
          "left_fixity_curves": ["ground_left_boundary"],
          "right_fixity_curves": ["ground_right_boundary"],
          "load_arrow_curves": ["kraan_load_p1", "kraan_load_p2", "water_loads"]
      }
"""

from .cli import main
from .geo_reader import eval_expr, parse_geo
from .mdpa_reader import parse_mdpa
from .models import GeoModel, MshModel
from .msh_reader import parse_msh
from .render import geo_to_svg, msh_to_svg

__all__ = [
    "GeoModel",
    "MshModel",
    "eval_expr",
    "geo_to_svg",
    "main",
    "msh_to_svg",
    "parse_geo",
    "parse_mdpa",
    "parse_msh",
]
