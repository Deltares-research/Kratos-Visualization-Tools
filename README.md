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

- [`visualize_mdpa_project.py`](./README_visualizer.md) — interactive MDPA mesh visualizer
- [`visualize_gid_results.py`](./visualize_gid_results.md) — GiD mesh and results visualizer
- [`plot_saved_results.py`](./plot_saved_results.md) — plot saved Kratos results

All three implementations are organized under `src/kratos_visualization_tools/`:
`mdpa.py` parses mesh data, `project.py` reads project parameters,
`visualization.py` provides the interactive UI, and `cli.py` handles command-line
execution. The `gid/` subpackage separates GiD parsing, result loading,
calculations, exports, drawing, viewers, settings, and CLI dispatch.
The `saved_results/` subpackage separates saved-data readers, series building,
plotting primitives, figure assembly, the viewer, configuration, and CLI dispatch.

The root scripts remain compatibility entry points, including their Python
import APIs and module-level monkeypatch points. After installation, the other
two tools can also be run without the root scripts:

```bash
kratos-visualize-gid path/to/case --list
kratos-plot-saved-results my_config.json --output-dir ./figures
```

## Optional GiD desktop interface

Install the PySide6 interface and launch it with `--gui`:

```powershell
python -m pip install -e ".[gui]"
python visualize_gid_results.py path\to\case --gui
```

The desktop interface embeds the existing Matplotlib plot and provides visible
controls instead of requiring hotkeys. It supports single-solution viewing and
folder browsing (`--folder path\to\results --gui`). The existing keyboard-driven
viewer remains the default; comparison mode continues to use that viewer.
See the [GiD documentation](./visualize_gid_results.md) for details.

## Tests

```bash
python -m pytest
```
