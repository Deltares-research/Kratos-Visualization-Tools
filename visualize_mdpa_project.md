====================================================================

⚠️  AI-ASSISTED PROTOTYPE — PROVENANCE & VALIDATION NOTICE

====================================================================

Author: Gennady Markelov

This contribution (scripts and documentation) was generated with AI assistance and reviewed on selected internal test cases. It is provided as a prototype and should not be considered production-ready.
Please perform independent review, validation, and testing before use.
 
====================================================================

# MDPA Visualizer

A small interactive tool to visualize node-based SubModelParts from a Kratos `.mdpa` and to show processes referenced in `ProjectParameters.json`.

## Quick start

```bash
python visualize_mdpa_project.py path/to/your.mdpa -p ProjectParameters.json
```

When the project file contains an `ImportMDPAModeler`, the `.mdpa` path can be
read from its `Parameters.input_filename` value:

```bash
python visualize_mdpa_project.py -p ProjectParameters.json
```

With an explicit `.mdpa` path, a relative `-p/--project` path is resolved
relative to the `.mdpa` file directory. With project-only invocation, the
project path is resolved relative to the current directory and the referenced
`.mdpa` file is looked up beside the project file.

You can pass multiple project/orchestrator files:

```bash
python visualize_mdpa_project.py path/to/your.mdpa -p ProjectParameters.json orchestrator.json
```

You can also pass a directory to `-p`; the tool scans for likely stage/project JSON files there.

If `-p` is omitted, the tool auto-detects common JSON files in the `.mdpa` directory.

## Features

**Lightweight matplotlib version** (`visualize_mdpa_project.py`):
- Fast rendering, minimal dependencies
- Right-side SubModelPart panel with checkboxes and color swatches
- High-contrast color assignment with co-occurrence-aware separation
- Segmented marker rendering for nodes in multiple SubModelParts
- Coincident-node depth stacking (same XY location) for clearer overlap visualization
- Live `Find` filter that keeps names visible and highlights matches
- Bulk visibility controls:
	- `Show` / `Hide` apply to current `Find` matches
	- `Invert` applies globally (visible <-> hidden for all SubModelParts)
- Scrollable SubModelPart list and stage list (mouse wheel + sliders)
- Stage selector (`All` + discovered stages) that applies stage-active visibility directly to checkboxes
- Auto-discovery and expansion of project/stage JSON inputs
- `Draw mesh` toggle for black Element/Condition connectivity lines
- Kratos higher-order mesh topology handling:
	- `*2D3N`: corner-corner-midpoint
	- `SmallStrainUPwDiffOrderElement2D6N`: quadratic triangle boundary
	- `SmallStrainUPwDiffOrderElement2D8N`: quadratic quadrilateral boundary
	- `*2Plus2N` and `*3Plus3N`: separate linear and quadratic interface sides
- `Draw normals` toggle for active/shown line Conditions and boundary Elements
- Two-sided normals for `2Plus2N` and `3Plus3N` interface Elements
- Unique SubModelPart colors for normal vectors
- `Normal +` and `Normal -` controls to change normal-vector length
- `Aspect: free` / `Aspect: equal` toggle; `free` fits the geometry to the
	plotting area, while `equal` uses the same screen scale for X and Y so
	normals appear perpendicular to the elements
- Click on nodes to inspect memberships and related processes (grouped per SubModelPart)
- Coincident-node membership summary in node reports
- Stage-aware process reporting (`[stage] process`) with one stage per line
- Stage solver usage extraction for:
	- `problem_domain_sub_model_part_list`
	- `body_domain_sub_model_part_list`
	- duplicate `body_domain` labels are suppressed when identical to `problem_domain`
- Dedicated report window with:
	- scrollable text
	- wrapped long lines for readability
	- `Copy` to clipboard
	- `Export` to timestamped text file near the input `.mdpa`
- Report contents include:
	- selected node coordinates, coincident node IDs, and SubModelPart membership
	- the number of nodes in each listed SubModelPart
	- processes, operations, and modelers associated with each SubModelPart
	- `Used as problem_domain`: the SubModelPart is listed in the solver's
	  `problem_domain_sub_model_part_list`
	- `Used as body_domain`: the SubModelPart is listed in the solver's
	  `body_domain_sub_model_part_list`.This is omitted when it contains the same information as `Used as problem_domain`, which is common and keeps the report more compact.
	- `Used by processes`: the SubModelPart is listed in the solver's
	  `processes_sub_model_part_list`
- Node clicks open the report in a separate window while keeping the main plot open
- Mouse/touchpad interactions:
	- left-button drag panning
	- wheel/touchpad scroll cursor-centered zoom
- Progressive zoom workflow:
	- click a node to select it
	- press `z` to zoom near that node
	- press `z` repeatedly to zoom further
- Press `r` to reset view
- Marker size controls in the right panel (`point size +` and `point size -`), with stable on-screen marker size during zoom/pan
- `Save plot` saves only the plotting area as a timestamped PNG next to the input `.mdpa`

## Notes

- The lite version uses matplotlib (lightweight, batteries-included on most systems)
- The control panel is intentionally outside the plotting area for readability
- Mesh lines and normals follow current SubModelPart visibility; hidden or inactive
	SubModelParts are excluded from the corresponding drawing
- Keyboard shortcuts in lite version:
	- `1..9, 0` show process report for SubModelPart #1..#10
	- `z` zooms to the last clicked node (progressive)
	- `r` resets zoom

