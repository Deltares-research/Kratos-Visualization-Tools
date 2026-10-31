====================================================================

⚠️  AI-ASSISTED PROTOTYPE — PROVENANCE & VALIDATION NOTICE

====================================================================

Author: Gennady Markelov

This contribution (scripts and documentation) was generated with AI assistance and reviewed on selected internal test cases. It is provided as a prototype and should not be considered production-ready.
Please perform independent review, validation, and testing before use.

====================================================================

# Kratos Visualization Tools

This repository contains lightweight visualization utilities for Kratos
Multiphysics meshes and simulation output: an MDPA mesh visualizer, a GiD results
viewer, and a configuration-driven plotter for saved results.

## Install

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
```

## Run the MDPA visualizer

```bash
python visualize_mdpa_project.py path/to/model.mdpa -p ProjectParameters.json
```

After installation, the equivalent console command is
`kratos-visualize-mdpa`. To use project-only mode, pass `-p` with a
`ProjectParameters.json` containing an `ImportMDPAModeler`.

Legacy scripts and their documentation remain at the repository root.

- [`visualize_mdpa_project.py`](./visualize_mdpa_project.md) — interactive MDPA mesh visualizer
- [`visualize_gid_results.py`](./visualize_gid_results.md) — GiD mesh and results visualizer
- [`plot_saved_results.py`](./plot_saved_results.md) — plot saved Kratos results

All three implementations are organized under `src/kratos_visualization_tools/`:
`mdpa.py` parses mesh data, `project.py` reads project parameters,
`visualization.py` provides the interactive UI, and `cli.py` handles command-line
execution. The `gid/` subpackage separates GiD parsing, result loading,
calculations, exports, drawing, viewers, settings, and CLI dispatch.
The `saved_results/` subpackage separates saved-data readers, series building,
plotting primitives, figure assembly, the viewer, configuration, and CLI dispatch.

The MDPA viewer includes display-only **Separated Faces** with adjustable gap,
node-ID lookup and local connectivity labels, plus stage activation filtering.
See the [MDPA guide](./visualize_mdpa_project.md) for geometry support and
separation diagnostics. Original MDPA coordinates are never modified.

The root scripts remain compatibility entry points, including their Python
import APIs and module-level monkeypatch points. After installation, the other
two tools can also be run without the root scripts:

```bash
kratos-visualize-gid path/to/case --list
kratos-plot-saved-results my_config.json --output-dir ./figures
```

## Windows launchers

The three `.bat` launchers live beside their matching Python scripts:

| Launcher | Input |
|---|---|
| [`visualize_mdpa_project.bat`](./visualize_mdpa_project.bat) | MDPA mesh |
| [`visualize_gid_results.bat`](./visualize_gid_results.bat) | GiD mesh, results file, or case base path |
| [`plot_saved_results.bat`](./plot_saved_results.bat) | Plot configuration JSON |

Drag an input file onto its launcher in File Explorer, or double-click the launcher
and paste or drag the input path into the prompt, then press Enter. The MDPA
launcher auto-detects project JSON files beside the mesh when `-p` is omitted.
The GiD launcher opens the default Matplotlib viewer; use `--gui` for the optional
desktop interface.

The launchers use `.venv\Scripts\python.exe` in this repository when present,
otherwise `python` from PATH. Install the dependencies in that environment first.
Keep the launchers beside the scripts; create Windows shortcuts rather than
copying the batch files into model folders.

All command-line arguments are forwarded unchanged, so options remain available:

```powershell
.\visualize_mdpa_project.bat "C:\Models\my case\model.mdpa" -p ProjectParameters.json
.\visualize_mdpa_project.bat -p "C:\Models\my case\ProjectParameters.json"
.\visualize_gid_results.bat "C:\Models\my case\case.post.res" --gui
.\plot_saved_results.bat "C:\Models\my case\plots.json" --show
```

The caller's working directory is preserved, so relative paths behave as they do
with the Python commands. Paths containing spaces must be quoted on the command
line. Failures pause the window so you can read the error, and the launchers
preserve the Python process's exit code.

## Optional GiD desktop interface

Install the PySide6 interface and launch it with `--gui`:

```powershell
python -m pip install -e ".[gui]"
python visualize_gid_results.py path\to\case --gui

# Open the three-button workflow chooser
python visualize_gid_results.py --gui

# Compare two solutions with desktop controls
python visualize_gid_results.py path\to\first --compare path\to\second --gui
```

The desktop interface embeds the existing Matplotlib plot and provides visible
controls instead of requiring hotkeys. It supports single-solution viewing,
folder browsing (`--folder path\to\results --gui`), and two-solution comparison
(`--compare path\to\second --gui`; `--comparison` is an alias).
Run with `--gui` and no file/folder to open a chooser for those three workflows.
The existing keyboard-driven viewer remains the default without `--gui`.
See the [GiD documentation](./visualize_gid_results.md) for details.

## Tests

```bash
python -m pytest
```
