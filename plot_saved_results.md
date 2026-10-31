# `plot_saved_results.py`

Standalone plotter for GeoMechanics validation results.

It reproduces the figures made by `test_CROW_case.py`, `eemdijk_blue_case.py`,
`eemdijk_green_case.py` and `retention_law_plotter.py`, but works purely on data that is
already on disk: Kratos JSON output files, comparison CSV/text files and (optionally) an
`.mdpa` file for the nodal coordinates.

No Kratos import, no simulation run — only `matplotlib` is required. What is plotted is
described entirely by a JSON input file, so new plots need no code changes.

After installation, `kratos-plot-saved-results` accepts the same arguments as
`python plot_saved_results.py`. The implementation lives in
`kratos_visualization_tools.saved_results`; the root module retains its existing
import API.

## Usage

On Windows, drag a plot configuration JSON file onto `plot_saved_results.bat`,
or double-click the launcher and supply the file at its prompt. It also accepts
the same arguments as the Python script, including `--show`.
See [Windows launchers](./README.md#windows-launchers) for details.

```powershell
python plot_saved_results.py plot_configs\crow_staged_construction.json
python plot_saved_results.py my_config.json --output-dir .\figures
python plot_saved_results.py my_config.json --show
python plot_saved_results.py --print-schema unused.json
```

`--show` opens a window instead of writing files. Its buttons let you view the figures
**All together** or **One by one** (with **< Previous** / **Next >**), and **Save** writes
all figures to the output directory.

From Python:

```python
import plot_saved_results

plot_saved_results.run(config_path, output_dir_override=case_dir)
```

## Profiles from the GiD desktop viewer

The GiD GUI's **Open in plotter** action automatically exports the current
profile CSV, creates this plotter's JSON configuration, and opens its viewer.
Comparison mode exports both profiles; **Plot + diff** also includes
their overlapping difference in a separate figure.

Keep `profile.csv` and `plot_config.json` together. All plotting paths are
relative to the bundle, so it can be moved or shared and reopened without the
original GiD files. The configuration's `metadata` records the original sources,
result, actual steps, component/reduction, and profile geometry; it does not
require those sources to exist. See the
[GiD profile handoff](./visualize_gid_results.md#send-the-current-gui-profile-to-the-plotter)
for usage.

## Input validation

Missing data files or columns produce a warning and the corresponding series is skipped;
a figure without any data is skipped entirely. Configuration errors are reported as a
single-line message and yield exit code 1.

Example configurations live in `plot_configs/`:

| File | Reproduces |
|------|------------|
| `crow_staged_construction.json` | wall and interface plots of `test_CROW_case.py` |
| `eemdijk_blue.json` | plots of `eemdijk_blue_case.py` |
| `eemdijk_green.json` | plots of `eemdijk_green_case.py` |
| `retention_law_characteristics.json` | characteristics plot of `retention_law_plotter.py` |

## Input file

```jsonc
{
  "data_dir": ".",            // where the data lives, relative to the config file
  "output_dir": "plots",      // where the figures go, relative to the config file
  "mesh": { "mdpa_file": "common/model.mdpa" },
  "node_sets": {
    "wall":   { "sub_model_part": "Sheet_Pile_Wall" },
    "probes": { "ids": [12, 34, 56] }
  },
  "stages": "auto",        // or an explicit list of stage names
  "figures": [ /* see below */ ]
}
```

`mesh` is only needed when a series uses nodal coordinates or a `node_set`. Instead of an
`mdpa_file` the coordinates may be listed inline:

```jsonc
"mesh": { "node_coordinates": { "12": [0.0, -1.5], "34": [0.0, -2.0] } }
```

Keys starting with `//` are ignored and can be used for comments.

Set top-level `stages` to `"auto"` to discover stage names from existing files
whose series `file` patterns contain `{stage}`. The plotter combines matching
stages across those patterns, removes duplicates, and sorts stage names
naturally, so numbered stages appear in numeric order. If no matching files are
found, configuration loading fails with an explanatory error. An explicit
stage-name list remains supported.

### Reusable templates

Use `figure_defaults` for options shared by every figure, `figure_templates` for
named groups of figure options, and `series_templates` for reusable series
options. A figure selects a named template with `"template"`; a series does the
same inside its `series` list. Values merge in this order: figure defaults,
figure template, then explicit figure options. For a series, explicit series
options override the selected series template. The `template` reference itself
is removed before plotting.

```jsonc
{
  "figure_defaults": { "ylabel": "$y$ [m]" },
  "figure_templates": {
    "wall_history": { "type": "histories_per_node", "node_set": "wall" }
  },
  "series_templates": {
    "kratos": { "label": "Kratos", "marker": "." }
  },
  "figures": [
    {
      "template": "wall_history",
      "name": "pressure.svg",
      "series": [
        { "template": "kratos", "file": "{stage}.json", "variable": "PRESSURE" }
      ]
    }
  ]
}
```

Unknown template names are reported as configuration errors. Templates are
shallow merges: nested objects and lists are replaced, not recursively merged.

### Figures

Every figure has a `name` (the output file), a `type` and a list of `series`. The stage
list defaults to the top-level `stages`.

| Type | Result |
|------|--------|
| `profiles_per_stage` | one subplot per stage, all sharing the same series definitions |
| `histories_per_node` | one subplot per node of `node_set`/`node_ids`, x is usually the stage time |
| `single` | one axes; with `"per_stage_files": true` one file per stage |
| `custom_subplots` | subplots with individual titles, labels and series (`subplots` list) |

Common figure options:

| Option | Default | Meaning |
|--------|---------|---------|
| `stages` | top-level `stages` | stages used for this figure |
| `title` | stage name / `Node <id>` | subplot title (`single` and `custom_subplots`) |
| `xlabel`, `ylabel` | — | axis labels; `ylabel` is drawn on the first column only |
| `xscale`, `yscale` | linear | e.g. `"log"` |
| `log_y` | `false` | shorthand for `"yscale": "log"` |
| `invert_yaxis` | `false` | invert the y axis |
| `max_plots_per_row` | `5` | subplot layout |
| `figsize` | `[20, 6]` | figure size in inches |
| `per_stage_files` | `false` | `single` only: write one file per stage |
| `annotate_min_max` | `false` | `single` only: add a min/max box per series |
| `node_set`, `node_ids` | — | `histories_per_node` only: the nodes to plot |

### Series

Each series reads either a Kratos JSON output file or a CSV/text file.

Common options: `label`, `line_style` (default `"-"`), `marker`, `color`,
`sort_by` (`"x"`, `"y"` or omitted), and `expand_over_stages` (default `false`).

For a combined `single` figure, set `expand_over_stages` to `true` on a series template
to create one curve per stage in the figure's stage list. Use `{stage}` in the label and
data options such as `filter` to distinguish the curves. This does not apply when
`per_stage_files` is enabled.

**`"source": "kratos_json"`** (the default)

| Option | Default | Meaning |
|--------|---------|---------|
| `file` | — | file name, may contain `{stage}` |
| `node_set` / `node_ids` | — | nodes to read (not needed in `histories_per_node`) |
| `variable` | — | e.g. `BENDING_MOMENT`, `GEO_EFFECTIVE_TRACTION_VECTOR` |
| `component` | — | component of a vector-valued variable |
| `time_index` | `0` | index into the output-step list of the JSON file |
| `scale`, `offset` | `1.0`, `0.0` | applied as `scale * value + offset` |
| `x`, `y` | `"value"`, `"node_y"` | one of `value`, `node_x`, `node_y`, `node_z`, `node_id`, `time` |
| `skip_zero_values` | `true` | histories only: drop exact zeros of not-yet-activated nodes |
| `over` | `"stages"` | histories only: set to `"nodes"` to plot a profile inside a per-node figure |

In `histories_per_node` the series is built by reading one file per stage and taking the
value of the current node, giving `(time, value)` points.

**`"source": "csv"`**

| Option | Default | Meaning |
|--------|---------|---------|
| `file` | — | file name, may contain `{stage}` |
| `x_column`, `y_column` | `0`, `1` | column name (needs a header) or zero-based index; may contain `{stage}` and `{node}` |
| `x_scale`, `y_scale` | `1.0` | multiplied with the column value |
| `x_offset`, `y_offset` | `0.0` | added after scaling |
| `missing_value` | `"N/A"` | rows with this entry are skipped |
| `comment_prefix` | `"#"` | lines starting with it are ignored |
| `filter` | — | `{column: value}`; only rows where every listed column equals its value are used; values may contain `{stage}` and `{node}` |

Comma-separated and whitespace-separated files are both supported; a header row is
detected automatically (a first row that cannot be parsed as numbers).

### Placeholders

`{stage}` and `{node}` are substituted in figure names, titles, file names and column
names, so one definition covers all stages and nodes.

## Example

Bending moments of a sheet pile wall per stage, Kratos against a commercial FE package:

```jsonc
{
  "data_dir": "../crow_validation",
  "output_dir": "../crow_validation/plots",
  "mesh": { "mdpa_file": "common/model.mdpa" },
  "node_sets": { "wall": { "sub_model_part": "Sheet_Pile_Wall" } },
  "stages": ["3_Wall_installation", "4_First_excavation"],
  "figures": [
    {
      "name": "bending_moments.svg",
      "type": "profiles_per_stage",
      "xlabel": "Bending moment [kNm/m]",
      "ylabel": "$y$ [m]",
      "series": [
        {
          "label": "Kratos",
          "file": "linear_elastic/staged_construction/as-is/{stage}__output_wall.json",
          "node_set": "wall",
          "variable": "BENDING_MOMENT",
          "scale": 0.001,
          "marker": "."
        },
        {
          "source": "csv",
          "label": "Commercial FE package",
          "file": "linear_elastic/staged_construction/{stage}__FE_comparison_wall.csv",
          "x_column": 1,
          "y_column": 0,
          "x_scale": -1.0,
          "line_style": "",
          "marker": "3"
        }
      ]
    }
  ]
}
```

## Tests

`test_plot_saved_results.py` covers the `.mdpa` and table readers, the series builders and
the figure generation. It needs no Kratos build:

```powershell
python -m pytest test_plot_saved_results.py
```
