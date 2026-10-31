# `visualize_gid_results.py`

Standalone viewer and extractor for GiD mesh and result files. The script reads a GiD
`.post.msh` mesh together with a matching `.post.res` result file and can:

- list available result names, locations and steps;
- display scalar, vector or matrix results on the mesh;
- compare two solutions with different meshes;
- show a vertical profile through the mesh;
- save a static plot or open an interactive viewer;
- export values sampled along an arbitrary line to CSV.

The default script only needs `numpy` and `matplotlib`; it does not run a Kratos
simulation. The optional desktop interface additionally requires `PySide6`.

After installation, `kratos-visualize-gid` accepts the same arguments as
`python visualize_gid_results.py`. The implementation lives in
`kratos_visualization_tools.gid`; the root module retains its existing import API.

## Optional desktop interface

Install the optional GUI dependencies from the repository root:

```powershell
python -m pip install -e ".[gui]"

# Open one solution with visible desktop controls
python visualize_gid_results.py path\to\case --gui

# Browse the solutions in a folder
python visualize_gid_results.py --folder path\to\results --gui

# Compare two solutions with visible desktop controls
python visualize_gid_results.py path\to\first --compare path\to\second --gui

# Reopen saved comparison settings in the desktop interface
python visualize_gid_results.py --settings comparison_PRESSURE_settings.json --gui
```

After installation, `kratos-visualize-gid` also accepts `--gui`.
The PySide6 window embeds the Matplotlib plot with a navigation toolbar and a
control sidebar. Result, step, location, and component selectors replace
keyboard-only browsing. Display controls expose mesh, gradients, isolines,
labels, profiles, and displacement deformation. Image/settings and CSV exports
are available through visible controls, and coordinate inspection displays a
desktop report. Previous/next controls and playback make time-step browsing
discoverable.

Existing initial-view options, such as `--result`, `--component`, `--profile`,
`--profile-path`, and `--settings`, can be used with `--gui`.
The GUI is opt-in for single solutions, folder browsing, and two-solution
comparison. `--comparison` is an alias for `--compare`. Without `--gui`, the
existing keyboard-driven viewers remain the default. Listing, batch extraction,
and static exports remain available without PySide6. `--gui` cannot be combined
with `--list`, `--extract-json`, or static `--save` (unless `--interactive` is
also supplied); use the desktop Save image button for interactive exports.
Folder browsing cannot be combined with comparison.

The comparison sidebar provides:

- common result and location selectors, vector components and tensor reductions;
- `latest` (resolved independently for each solution), common numeric steps,
  a step slider, and Previous/Next step buttons;
- all five comparison styles, colormap, contour levels, isolines and labels;
- mesh, domain-edge, and profile toggles, with editable profile x/path;
- an editable shared color range and an Automatic color range button;
- nearest-node or IDW coordinate inspection by click or typed `x,y`;
- image and comparison-settings saves.

The comparison window uses the same field loading, interpolation, difference,
profile, and inspection calculations as the existing comparison viewer.
Its Matplotlib toolbar supplies pan/zoom without triggering coordinate inspection.
Single-solution color ranges remain automatic; `--vmin` and `--vmax` are
accepted with `--gui` only in comparison mode.

## Input files

The script accepts a base path, a mesh path or a result path. It recognizes these pairs:

| Input | Matching files |
|---|---|
| `case` | `case.post.msh` and `case.post.res` (preferred), or `case.msh` and `case.res` |
| `case.post.msh` | `case.post.msh` and `case.post.res` |
| `case.post.res` | `case.post.msh` and `case.post.res` |

Mesh blocks with `Triangle` and `Quadrilateral` elements are used for the surface plot.
`Linear` mesh blocks are used to detect vertical profile lines and to locate line-based
Gauss-point results.

## Usage

On Windows, drag a mesh or results file onto `visualize_gid_results.bat`, or
double-click the launcher and supply the file at its prompt. It also accepts
the same arguments as the Python script, including `--gui`.
See [Windows launchers](./README.md#windows-launchers) for details.

Run these commands from this directory, or provide paths relative to the current directory:

```powershell
# List result blocks and available steps
python visualize_gid_results.py path\to\case --list

# Open the interactive viewer for the latest water pressure result
python visualize_gid_results.py path\to\case --result WATER_PRESSURE --step latest

# Save a static plot instead of opening the viewer
python visualize_gid_results.py path\to\case `
    --result REACTION --component magnitude --step latest `
    --save figures\reaction.png

# Save a scalar field with solid isolines over the filled field
python visualize_gid_results.py path\to\case `
  --result WATER_PRESSURE --isolines --isoline-labels --contour-levels 20 `
  --save figures\water_pressure_isolines.png

# Include the vertical profile panel
python visualize_gid_results.py path\to\case `
  --result WATER_PRESSURE --profile `
  --save figures\water_pressure_with_profile.png
```

When `--save` is omitted, the interactive viewer is used. The viewer supports changing
results and steps, vector components, tensor reductions, color maps, mesh visibility,
zooming, panning, solid isolines and CSV export through its keyboard and mouse controls.
Use `--isolines` to enable isolines initially; press `l` in the interactive viewer to
toggle them. Isolines are drawn with explicit solid black lines over triangulated scalar
or reduced matrix fields. Use `--contour-levels` to change the number of levels; the
default is 12 and the minimum is 2. Use `--isoline-labels` to show compact inline values
on the isolines; press `L` interactively to toggle them. Press `c` to cycle the colormap, `s` to save the current
interactive figure as a PNG, and `S`/`Shift+S` to save the current visualization settings
as JSON. The `p` key remains available for the previous-result action. The script
disables Matplotlib's native `s` save-dialog and `l` logarithmic-Y shortcuts to avoid
conflicts. The keyboard and state menu panels can be dragged to other locations with
the mouse.

The profile panel is hidden by default so the field plot uses the available window area.
Use `--profile` to enable the vertical profile panel in either single-solution or
comparison mode. In comparison mode, the profile layout contains the two solution
profiles and a separate profile-difference subplot. The difference is calculated as
solution 2 minus solution 1 over their overlapping profile coordinates.

## Comparing two solutions

Pass the first solution as the positional argument and the second with `--compare`:

```powershell
python visualize_gid_results.py path\to\solution_1 `
  --compare path\to\solution_2 `
  --result WATER_PRESSURE

# Save a comparison image with an explicit common scale
python visualize_gid_results.py path\to\solution_1 `
  --compare path\to\solution_2 `
  --comparison-style background+isolines `
  --inspection-method idw `
  --colormap cividis `
  --vmin -10000 --vmax 50000 `
  --save figures\water_pressure_comparison.png
```

Each solution is loaded using its own mesh and result file. At startup and in the viewer
menu, the script reports:

- variables available in both solutions, which are the only variables offered for comparison;
- variables unique to the first solution, which are not shown;
- variables unique to the second solution, which are not shown.

The result selected by `--result` must be common to both solutions. If it is not, the
first common result is selected and reported. With `--step latest`, each solution uses
its own latest available step. If the selected result is stored at different locations,
each solution uses the location available for that solution.

### Overlay styles

Use `--comparison-style` to select the initial overlay:

| Style | First solution | Second solution |
|---|---|---|
| `background+isolines` | filled colors in the background | black isolines on top |
| `isolines+background` | black isolines | filled colors in the background |
| `both-filled` | translucent filled colors | translucent filled colors |
| `both-isolines` | blue isolines | red isolines |
| `difference` | interpolated onto solution 2 mesh | `solution 2 - solution 1` |

Press `o` in the interactive viewer to cycle through these styles. All styles use one
shared color scale calculated from finite values in both solutions. Difference mode instead
interpolates solution 1 onto solution 2's mesh using inverse-distance weighting and plots
`solution 2 - solution 1` on solution 2's mesh. Its automatic scale is centered on zero,
so red and blue show opposite-signed differences. Set `--vmin` and `--vmax` to define the
scale explicitly. In interactive mode, press `[` or `]` to narrow or
widen the shared range and `a` to restore automatic scaling. Press `m` to show or hide
both meshes; solution 1 uses solid blue lines and solution 2 uses dashed red lines.
Press `v` to enter explicit shared limits as `vmin,vmax`; the values must be finite and
`vmin < vmax`. The limits are applied immediately to the comparison plot.
The outer domain boundaries are shown in grey by default: solution 1 is solid grey and
solution 2 is dashed grey. Press `b` to toggle these boundaries independently of the
full mesh. For saved comparison plots, use `--no-domain-edge` to hide them.
Use `--colormap` to select a comparison color map. The available maps are `viridis`,
`plasma`, `inferno`, `magma`, `cividis`, `YlGnBu`, `YlOrRd`, `coolwarm`, `RdBu_r`,
`seismic`, `Spectral` and `turbo`. Difference mode defaults to `RdBu_r`; other filled
comparison modes default to `viridis`. Press `c` interactively to cycle through them.
Press `s` to save the current comparison figure as a PNG, and `S`/`Shift+S` to save
the comparison settings as JSON; `p` remains the previous-result action.
Left-click the plotting area
to open a comparison table. The table finds the nearest node independently on each mesh
and reports each variable's value, node ID, step and location for both solutions. A `-`
marks a variable that is unavailable in that solution. Press `i` to enter an `x,y`
coordinate instead of clicking; the same comparison table is opened for that coordinate.
By default, each solution reports the value at its nearest node. Use
`--inspection-method idw` to use inverse-distance weighting of up to eight nearby valid
nodes instead. Exact node coordinates still return the node's exact value. Press `w` in
the interactive viewer to switch between `nearest` and `idw`.

## Visualization options

| Option | Default | Meaning |
|---|---|---|
| `--result` | `WATER_PRESSURE` | Result block to display |
| `--step` | `latest` | Numeric step or the latest available step |
| `--location` | automatic | `OnNodes` or `OnGaussPoints` |
| `--component` | `magnitude` | Vector component: `x`, `y`, `z` or `magnitude` |
| `--tensor-component` | `trace` | Matrix reduction: `xx`, `xy`, `yx`, `yy`, `trace` or `magnitude` |
| `--profile-x` | `18.0` | X coordinate used by profile fallback detection |
| `--x-tol` | `0.35` | Tolerance used for vertical-line/profile detection |
| `--profile` | off | Show the vertical profile panel |
| `--active` | empty | SubModelPart to display exclusively |
| `--mdpa` | automatic | `.mdpa` file used with `--active` |
| `--save` | empty | Static image output path; enables non-interactive mode |
| `--interactive` | false | Force interactive mode even when `--save` is supplied |
| `--colormap` | automatic | Comparison color map; difference defaults to `RdBu_r` |
| `--no-domain-edge` | off | Hide grey comparison-domain boundaries |

For `--active`, the script reads node, element and condition IDs from the selected MDPA
SubModelPart. Supply `--mdpa` when the file cannot be found automatically next to the
requested case.

## Line extraction

Use `--extract-json` to sample one or more result files along a line and write a CSV:

```powershell
python visualize_gid_results.py `
    --extract-json extract_pressure.json `
    --csv-out output\pressure_profile.csv
```

The JSON must contain a non-empty `files` array and line coordinates. It may be either a
top-level configuration object or an object with an `extract` object:

```json
{
  "files": ["case_01", "case_02.post.msh"],
  "result": "WATER_PRESSURE",
  "step": "latest",
  "location": "auto",
  "component": "magnitude",
  "line": {
    "start": [18.0, -12.0],
    "end": [18.0, 4.0],
    "samples": 200
  },
  "output_csv": "pressure_profile.csv"
}
```

`line_points: [[x0, y0], [x1, y1]]` may be used instead of `line.start` and `line.end`.
The sample count can also be specified as the top-level `line_samples`. Relative file
paths and `output_csv` paths are resolved relative to the JSON file.

For `OnGaussPoints` data, element values are projected to nodes before line sampling.
Sampling uses inverse-distance weighting and preserves exact values when a sample lands
on a node.

The generated CSV contains these columns:

| Column | Meaning |
|---|---|
| `file_base` | Input entry from `files` |
| `result`, `step`, `location`, `component` | Selection used for that file |
| `sample_index` | Zero-based sample number |
| `t`, `s` | Normalized position and distance along the line |
| `x`, `y` | Sample coordinates |
| `value` | Interpolated result value |
| `gradient` | Finite-difference derivative with respect to `s` |

### Plotting the extracted CSV

Add `--plot-config` to plot the CSV with `plot_saved_results.py` right after it is
written; add `--plot-show` to open its viewer window instead of saving the figures:

```powershell
python visualize_gid_results.py `
    --extract-json extract_pressure.json `
    --plot-config plot_pressure.json --plot-show
```

Use the CSV `filter` option to draw one line per solution:

```json
{
  "figures": [
    {
      "name": "pressure_profile.svg",
      "type": "single",
      "xlabel": "Water pressure",
      "ylabel": "y [m]",
      "series": [
        {"source": "csv", "file": "pressure_profile.csv", "x_column": "value",
         "y_column": "y", "filter": {"file_base": "case_01"}, "label": "case_01"},
        {"source": "csv", "file": "pressure_profile.csv", "x_column": "value",
         "y_column": "y", "filter": {"file_base": "case_02.post.msh"}, "label": "case_02"}
      ]
    }
  ]
}
```

## Tests

The parser and extraction helpers are covered without requiring a Kratos simulation:

```powershell
python -m pytest test_visualize_gid_results.py -q
```