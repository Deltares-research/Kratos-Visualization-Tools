====================================================================

⚠️  AI-ASSISTED PROTOTYPE — PROVENANCE & VALIDATION NOTICE

====================================================================

Author: Gennady Markelov

This contribution (scripts and documentation) was generated with AI assistance and reviewed on selected internal test cases. It is provided as a prototype and should not be considered production-ready.
Please perform independent review, validation, and testing before use.

====================================================================

# `geo_to_svg.py`

`geo_to_svg.py` turns a 2D geotechnical model into a static SVG picture, such as a
cross-section for a report or presentation. It accepts three kinds of input:

| Input | Source | What is drawn |
| --- | --- | --- |
| `.geo` | Gmsh geometry script | Physical surfaces as patterned regions; physical curves as coloured lines |
| `.msh` | Gmsh mesh (format v2 or v4) | Each 2D mesh element, coloured by physical group; shared edges outlined |
| `.mdpa` | Kratos model part | SubModelPart elements as patterned regions; open element/condition paths and node-only SubModelParts as curves |

GEO and MDPA input produce the same kind of picture. You can add support and load
symbols, a phreatic line and a legend that you control from a JSON file.

The output is plain SVG. Each part of the picture is a named group, so you can edit it
in Inkscape or a browser, or convert it to PDF/PNG.

## Requirements

- GEO and MSH input need only the Python standard library.
- MDPA input reuses the package MDPA parser (`kratos_visualization_tools.mdpa`) and
  therefore needs the project dependencies (`numpy`). Install them with:

  ```bash
  python -m pip install -e .
  ```

  The script finds the package in `src/` next to the script, so you can also run it
  from a checkout without installing.

## Usage

```bash
python geo_to_svg.py --input model.geo  --output model.svg
python geo_to_svg.py --input model.msh  --output model_mesh.svg
python geo_to_svg.py --input model.mdpa --output model.svg
python geo_to_svg.py --input model.mdpa --output model.svg --legend-config legend.json
```

| Option | Default | Description |
| --- | --- | --- |
| `--input` | *(required)* | `.geo`, `.msh` or `.mdpa` file; the extension selects the reader |
| `--output` | *(required)* | SVG file to write; parent folders are created |
| `--width` | `2200` | SVG width in pixels |
| `--height` | `1200` | SVG height in pixels |
| `--padding` | `70` | Margin around the model in pixels |
| `--legend-config` | none | JSON file that controls the legend, symbols and phreatic line |

The model keeps its aspect ratio: it is scaled to fit inside the padded canvas and
centred. The Y axis points up, as in the model.

## Input details

### Gmsh `.geo`

The reader supports the entities common in 2D profiles:

- `Point`, `Line`, `Spline`
- `Curve Loop` / `Line Loop`
- `Plane Surface`
- `Physical Curve` / `Physical Line` / `Physical Surface` (quoted names)

Numeric expressions may use variables defined earlier in the file, `Pi`, the
operators `+ - * / **`, unary `+`/`-` and parentheses (Gmsh's `^` is not supported). Expressions are evaluated safely; no
arbitrary Python is run. Statements may span several lines. Unsupported statements
are ignored.

### Gmsh `.msh`

- ASCII format v2 and v4.
- Triangles and quadrilaterals, both linear and higher-order. Only corner nodes are
  drawn.
- Line elements are used to find physical curve names.
- Each element is filled with its physical-surface colour.
- Edges between elements with different styles are drawn in `shared_edge_color`.
  Outer boundaries use `boundary_edge_width`.
- The legend lists the element and group counts.

The script stops with an error if the file has no triangles or quadrilaterals.

### Kratos `.mdpa`

The MDPA reader converts the Kratos model into the same internal model as a GEO file:

- Nodes are projected onto the XY plane (Z is ignored).
- Each element or condition is turned into node paths, using the geometry rules of
  the MDPA viewer.
  - A **closed** path (first node equals last, at least 3 distinct nodes) becomes a
    filled region.
  - An **open** path becomes a curve.
- Group names come from the SubModelParts that list the element or condition.
  Nested SubModelParts are named `Parent/Child`; an entity belongs to the
  SubModelPart block that lists it. Entities in no SubModelPart are grouped by
  their Kratos type name (e.g. `UPwSmallStrainElement2D3N`).
- **Node-only SubModelParts** (nodes but no elements or conditions) become curves
  along the outer element edges whose two end nodes are both in the group, so
  boundary groups such as `Bottom` or `Left_Edge` are coloured and can carry the
  hatching, fixity and load symbols. Edges inside the mesh are not used.
- Edges shared by adjacent elements of the same group are hidden, so each
  SubModelPart appears as one region. Only each region's outer outline (and
  holes) is written, with points on straight runs (e.g. mid-side nodes) dropped,
  so the SVG size depends on the outline length rather than the element count
  and no seams show between elements. Each curve group is one path; coincident
  segments, such as both faces of an interface element, are drawn once.
- An element or condition that belongs to several SubModelParts (e.g. an
  `interface_aggregate` group) is drawn only once. It goes to the first group
  listed in the legend config; otherwise to the smallest group, so aggregate
  groups lose to their more specific parts. This also applies to GEO input.

Limitations:

- A node-only SubModelPart whose nodes do not cover an outer element edge (e.g. a
  single node or an internal line of nodes) is not drawn.
- An element or condition that references a missing node is reported as an error,
  for example `Elements 12 references missing nodes: [99]`.
- An unsupported geometry is reported as an error, for example
  `Conditions 5: <reason>`.
- A file without nodes, or without elements and conditions, is rejected.

## SVG structure

| Group id | Mode | Content |
| --- | --- | --- |
| `physical_surfaces` | GEO/MDPA | Filled and patterned regions |
| `physical_curves` | GEO/MDPA | Coloured curves |
| `supports_and_loads` | GEO/MDPA | Bottom hatching, side fixities, load arrows, phreatic line |
| `mesh_elements` | MSH | One filled polygon per element |
| `mesh_shared_edges` | MSH | Element edge outlines |
| `coordinate_rulers` | all | Left and bottom rulers with ticks in model coordinates |
| `axes_triad` | all | X/Y axis arrows in the bottom-left corner |
| `legend` | all | Legend box |

Colours come from a stable hash of the group name, so the same name always gets the
same colour, unless a legend entry sets its own `color`. Surfaces also get hatch or
dot patterns, which keep regions distinguishable in greyscale prints.

## Legend configuration

The `--legend-config` JSON can have the settings at the root or nested under a
`"legend"` key. Every key is optional.

```json
{
  "legend": {
    "top_left": [40, 40],
    "max_curves": 18,
    "max_surfaces": 18,
    "curves": [
      {"name": "kraan_load_p1", "label": "Crane load", "color": "#E34066"},
      {"name": "Interface", "show": false}
    ],
    "surfaces": [
      {"name": "Clay"},
      {"names": ["Sand_top", "Sand_bottom"], "label": "Sand"}
    ],
    "phreatic_line": {
      "label": "Phreatic line",
      "points": [[0.0, 2.0], [50.0, 1.5], [100.0, -0.5]]
    },
    "symbols": {
      "bottom_hatch_curves": ["Bottom"],
      "left_fixity_curves": ["ground_left_boundary"],
      "right_fixity_curves": ["ground_right_boundary"],
      "load_arrow_curves": ["kraan_load_p1", "kraan_load_p2", "water_loads"]
    },
    "msh": {
      "shared_edges_black": true,
      "shared_edge_color": "#000000",
      "shared_edge_width": 1.0,
      "boundary_edge_width": 0.8
    }
  }
}
```

| Key | Meaning |
| --- | --- |
| `curves` / `surfaces` | Legend entries, in display order. `conditions` and `elements` are accepted as aliases (used when `curves` / `surfaces` is absent). Each entry is a group name string or an object with `name` (one group) or `names` (several groups), an optional `label`, an optional `color` (`#RRGGBB`, used for the drawing and the legend; other values are rejected), and an optional `show` (`false` hides the entry). Grouped names share one colour and one legend row; grouped surfaces are also drawn as one region with a single outer border (e.g. per-stage SubModelParts joined into one soil layer). Groups that are not listed keep their own style and border and are not shown in the legend. Without this key, all groups are listed alphabetically. |
| `max_curves` / `max_surfaces` | Maximum number of legend rows per section (default 18). |
| `top_left` | Legend position in SVG pixels, as `[x, y]` or `{"x": x, "y": y}` (default `[20, 20]`). |
| `rulers` | `false` hides the coordinate rulers and the axis triad (default `true`). |
| `phreatic_line` | Dashed line through `points` given in model coordinates (at least 2 points), with its own legend row. |
| `symbols` | Curve names for each symbol type. The defaults are shown in the example above. Bottom curves get ground hatching; left and right curves get roller-support symbols; load curves get downward arrows joined by a top line. |
| `msh` | MSH-only edge styling. If `shared_edges_black` is `false`, edges use the element colour. |

Names are matched exactly, including case. They must match the GEO physical names,
the MSH physical names or the MDPA SubModelPart names.

### Example

[`examples/geo_to_svg_legend_eemdijk.json`](./examples/geo_to_svg_legend_eemdijk.json)
is a complete configuration for the Eemdijk dike model, using its MDPA SubModelPart
names:

```bash
python geo_to_svg.py --input ..\FEA-Tools\Eemdijk\Eemdijk_blue\eemdijk.mdpa --output model.svg --legend-config examples\geo_to_svg_legend_eemdijk.json
```

In that MDPA, `Bottom` and the side boundaries are node-only SubModelParts; they are
drawn along the outer element edges, so their hatching and fixity symbols appear in
MDPA mode too.

[`geo_to_svg_legend_levee.json`](./geo_to_svg_legend_levee.json) shows the
`conditions` / `elements` aliases with fixed entry colours.

## Tests

```bash
python -m pytest tests/test_geo_to_svg.py
python -m pytest tests/test_geo_to_svg.py --cov=geo_to_svg --cov-branch --cov-report=term-missing
```

Pass the module name (`geo_to_svg`), not the file name, to `--cov`. Coverage needs
`pytest-cov`, which is not part of the `test` extra; install it with
`python -m pip install pytest-cov`.
