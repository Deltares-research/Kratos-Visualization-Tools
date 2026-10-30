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

On Windows, drag an MDPA file onto `visualize_mdpa_project.bat`, or double-click
the launcher and supply the file at its prompt. It also accepts the same arguments
as the Python script. See [Windows launchers](./README.md#windows-launchers) for
environment selection and examples.

```bash
python visualize_mdpa_project.py path/to/your.mdpa -p ProjectParameters.json
```

When the project file contains an `ImportMDPAModeler`, the `.mdpa` path can be
taken from its `Parameters.input_filename` value and `.mdpa` input can be omitted:

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

If `-p` is omitted, the tool checks the `.mdpa` directory for `<mdpa-name>_stages_input.json`, `<mdpa-name>_stages.json`, `ProjectParameters.json`, and other `.json` files with `stage` in the name.

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
- Stage selector (`All` + discovered stages) filters the available SubModelParts
  for the selected stage without changing manual checkbox visibility
- Auto-discovery and expansion of project/stage JSON inputs
- `Draw mesh` toggle for black Element/Condition connectivity lines
- Kratos higher-order mesh topology handling:
	- Known `2D3N` line/beam types: corner-corner-midpoint
	- Known `2D3N` triangle types: closed three-corner boundary
	- `*2D4N`: closed four-corner boundary
	- `SmallStrainUPwDiffOrderElement2D6N`: quadratic triangle boundary
	- `SmallStrainUPwDiffOrderElement2D8N`: quadratic quadrilateral boundary
	- `*2Plus2N` and `*3Plus3N`: separate linear and quadratic interface sides
- Unknown geometry is not guessed; unsupported separation topology is reported
  in node diagnostics rather than preventing the node view from opening
- `Draw normals` toggle for active/shown line Conditions and boundary Elements
- Two-sided normals for `2Plus2N` and `3Plus3N` interface Elements
- Unique SubModelPart colors for normal vectors
- `Normal +` and `Normal -` controls to change normal-vector length
- `Aspect: free` / `Aspect: equal` toggle; `free` fits the geometry to the
	plotting area, while `equal` uses the same screen scale for X and Y so
	normals appear perpendicular to the elements
- `Visualize` cycles through all, activated-only, and deactivated-only
	SubModelParts for the selected stage. Activation is seeded by each stage's
	`problem_domain_sub_model_part_list` and process references in the selected
	stage (including boundary and load parts). Process-only references do not
	carry forward to stages that no longer use them. `ActivateModelPartOperation` and
	`DeactivateModelPartOperation` update each part's status in stage/operation
	order, with the latest operation determining its state; a process reference
	does not override explicit deactivation. For a selected
	stage, `all` includes its referenced SubModelParts and all modeled parts
	that have been active or deactivated through that stage, preserving prior
	deactivations even when omitted from a later stage's domain list;
	Every part in `all` is classified by its latest explicit operation,
	defaulting to activated: `activated = all - deactivated`. This includes
	parts referenced only by modelers or other stage operations.
	With stage `All`, the Visualize mode adds no activity filter. Manual
	checkbox visibility applies in every mode and is independent of stage
	selection. `Show` / `Hide` apply to parts matching the selected Visualize
	mode and `Find`; `Invert` still applies globally. Stage-selection messages
	report active/deactivated counts before manual visibility filtering.
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
- Viewer keyboard shortcuts are suspended while editing the `Gap`, `Node ID`,
  or `Find` text fields.
- Marker size controls in the right panel (`point size +` and `point size -`), with stable on-screen marker size during zoom/pan
- `Save plot` saves only the plotting area as a timestamped PNG next to the input `.mdpa`
- Separated faces and local connectivity
The bottom toolbar provides `Separated Faces`, `Gap -` and `Gap +` buttons
with an editable `Gap` field in model units, `Node ID`, and a `Local labels`
toggle. Type a positive finite gap value and press Enter to apply it directly;
typing alone does not recalculate the plot. Scientific notation is supported.
Invalid values are reported in the console and leave the applied gap unchanged.
Each `Gap +` click multiplies the gap by `1.1`; `Gap -` divides it by
`1.1`. There is no application-defined upper limit. The gap remains positive,
subject only to floating-point numeric limits; turn `Separated Faces` off
to restore the original node positions. Gap changes apply immediately when separation
is enabled and are retained while it is off.
- `node ID` and `Local labels`
Enter a node ID and press Enter to zoom to its incident elements and coincident counterparts. Clicking a node also
highlights its local connectivity. Node labels are IDs; element and condition
labels use `E` and `C` prefixes. The report lists full connectivity, entity
types, SubModelParts, original coordinates, and configured pair relationships.

`Separated Faces` separates coincident counterparts identified through
`2Plus2N` / `3Plus3N` interface connectivity or configured master/slave
SubModelParts. It does not treat every coincident node as an interface pair.
For interface-only pairs, the lower-ID reference corner stays fixed. Declared
master/slave roles take precedence over node-ID ordering: masters stay fixed
and coincident slaves move, even when a slave has a lower ID. For each visible interface, its
normal guides the choice of a noncoincident connected mesh edge along which
the duplicate can move. Non-interface edges are preferred; visible
interface-side edges are used when those are the only available connections.
The normal is oriented toward the duplicate's soil side using only adjacent
soil elements that share another node on that interface face. Soil elements
attached to a different face at a shared tip do not determine its orientation.
Triangle soil centers use their three corners; `2D4N` and `2D8N`
quadrilateral soil centers use their four corners. Conditions do not
contribute soil centers.
Quadratic corner directions use the adjacent midside-node segments. Equally
suitable neighbors are selected by lowest node ID, so offsets do not depend
on entity ordering.

When several visible interfaces share a duplicate, compatible movement
contributions are averaged without normalization. If interface opening
directions or their selected movement directions are more than 90 degrees
apart, the duplicate stays fixed instead. This includes opposite directions
whose average would be zero. Conflicting nodes are marked with red crosses
while separation is enabled, and their reports include
`Conflicting visible interface directions; node left fixed`.
Each interface entity contributes once, even if it belongs to multiple
SubModelParts. Hiding one of its memberships does not exclude it while
another membership remains visible.
Enabling separation recalculates offsets using current visibility; an interface
is eligible only when all its connected nodes are visible.

For quadratic interfaces, lines, triangles, and quadrilaterals, each midside
node receives the arithmetic average of its two corner movement vectors,
after applying clone ranks to the corners, including midside nodes that are
not themselves paired. Interpolated midsides do not receive another rank
multiplier. For example, corner offsets of `0.002` and `0.001` in the same
direction produce a midside offset of `0.0015`. Fixed corners contribute
zero. Hidden interfaces do not contribute corner pairs to this interpolation;
non-interface quadratic geometry still participates. A constraint slave whose
only usable interpolation geometry is a hidden interface stays fixed with a
missing-connectivity diagnostic. The average is not normalized; opposite corner vectors can cancel, leaving
the midside fixed. A midside node with its own conflicting interface directions
stays fixed rather than being moved by interpolation.
The base display offset is `0.01 * Gap`, giving finer separation control:
`Gap = 0.1` moves the first duplicate with a unit movement vector by `0.001`
model units. Averaged vectors can produce smaller offsets. For several clones
at one position, successive offsets are multiples of this reduced base offset.

Master/slave coincidence identifies candidates, not verified runtime
constraints. The project JSON fields `master_model_part_name` and
`slave_model_part_name` are read from process parameters and stage-preprocess
operation/modeler parameters. Nodes in the two referenced SubModelParts are
looked up using the original qualified names: exact MDPA keys take precedence,
then leading model-part prefixes are removed one at a time until a key matches
(for example, `Domain.Structure.Wall` resolves to `Structure.Wall` before
`Wall`). Resolved MDPA keys are used for pairing, role tracking, visibility,
and relationship reports. Unresolved names produce a console warning and the
relation is excluded; unrelated parts with the same leaf name are not guessed.
Nodes in the resolved SubModelParts are
paired by XYZ distance within `1e-8` model units; noncoincident nearest-neighbor
constraints are not reconstructed. A slave with multiple coincident masters,
or a node declared as both master and slave, stays fixed with a diagnostic.
Master candidates are counted by unique node ID across all considered
relations after stage and visibility filtering. Overlapping SubModelParts
or repeated relations referencing the same master node do not create
additional candidates; all relation metadata is retained.
Master nodes also stay fixed during quadratic midside interpolation.
Without interface geometry, attached mesh, line, beam, or boundary
connectivity still supplies movement edges. Nodes without a noncoincident
connected edge remain fixed, with diagnostics in the report and red crosses
in the inspected neighborhood.

Unsupported separation topology is skipped rather than preventing the node
view from opening, including when visibility changes rebuild separation data.
Affected nodes retain diagnostics identifying the entity block, ID, and type;
these appear in node reports and as red crosses in the inspected neighborhood.
Supported neighboring geometry still contributes to separation. If no usable
connected geometry remains, the candidate stays fixed. Unknown connectivity
is not guessed; this does not add mesh or normal rendering support for those
entity types.

Each node ID has one display position, shared by nodes, mesh lines, normals,
picking, and zoom. The MDPA and original coordinates are never changed. Turning
separation off restores the original coordinates and coincident-circle view.
Orange dashed links identify shown interface pairs; teal dashed links identify master/slave-only candidates. These are diagnostic
correspondence links, not additional mesh elements.
Interface links and offsets use the same entity-key visibility rule: all
connected nodes must be visible, and at least one entity membership must be
visible when memberships exist. Interfaces without entity memberships remain
eligible and produce orange links, including when a master/slave constraint
also exists. Reports list visible interface entity keys separately from their
SubModelPart names; an ungrouped interface can have visible entities but no
part names.

Checkbox changes, `Show` / `Hide` / `Invert`, and stage selection recalculate
offsets using currently visible interfaces and shown stage-referenced
master/slave candidates. A shared node can therefore stay fixed with both
conflicting interfaces visible, then move when only one is visible. With
neither visible, it has no interface-driven displacement; a visible
master/slave candidate can still supply a displacement.

The viewer uses a shared refresh sequence: synchronize checkbox visibility,
rebuild separation when visibility or stage changes (and when enabling
separation), update display coordinates, then redraw nodes, mesh, normals,
and diagnostics. Startup follows the same sequence. Gap changes reuse the
resolved movement vectors; layer-only controls do not rebuild separation.

For example, in the Eemdijk mesh, node `21332` is coincident with node `27`
and is shared by interfaces `6223` and `6403`. With both conflicting interfaces
visible, it should stay fixed. Showing only one interface resolves separation
using that interface's contribution.

Reports distinguish stage references from runtime activation: the existing
stage metadata does not establish whether an
interface or constraint is actually active in a running simulation. Selecting
a stage does not load a different MDPA; open that stage's MDPA separately if its
mesh differs.

For the bottom-plate example, enter node `48` (or clone `16160`), enable
`Separated Faces`, and try a gap of `0.03`. Use `Draw mesh` for the surrounding
mesh and `Draw normals` for boundary directions. Incident connectivity is
highlighted even when the full mesh is off; `Local labels` can reduce clutter.

Paired corner nodes and their interpolated quadratic midside nodes are displaced.
Attached elements therefore deform on screen, but there is no tapered interior mesh
deformation or guarantee against crossings/inverted displayed elements. Keep
the gap small; the separated view is not a physical opening or interface
thickness.

## Notes

- The lite version uses matplotlib (lightweight, batteries-included on most systems)
- The control panel is intentionally outside the plotting area for readability
- Mesh lines and normals follow current SubModelPart visibility; hidden or inactive
	SubModelParts are excluded from the corresponding drawing
- Keyboard shortcuts in lite version:
	- `1..9, 0` show process report for SubModelPart #1..#10
	- `z` zooms to the last clicked node (progressive)
	- `r` resets zoom
