"""Standalone plotter for GeoMechanics validation results.

Reproduces the figures that ``test_CROW_case.py``, ``eemdijk_blue_case.py``,
``eemdijk_green_case.py`` and ``retention_law_plotter.py`` produce, but works
purely on data that has already been written to disk (Kratos JSON output files,
comparison CSV files and an optional ``.mdpa`` file for the nodal coordinates).

It does not import Kratos, so no simulation is run and no Kratos installation is
required: only ``matplotlib`` is needed.

Usage::

    python plot_saved_results.py plot_configs/crow_staged_construction.json
    python plot_saved_results.py my_config.json --output-dir ./figures
    python plot_saved_results.py my_config.json --show

See ``plot_configs/`` for example input files and the module level constant
``CONFIG_DOC`` below for the accepted schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from .config import CONFIG_DOC, run
from .viewer import FigureViewer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path, help="JSON input file describing the plots")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory from the config file",
    )
    parser.add_argument(
        "--print-schema", action="store_true", help="Print the config schema and exit"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figures in a window instead of saving them; use its Save button to save",
    )
    args = parser.parse_args(argv)

    if args.print_schema:
        print(CONFIG_DOC)
        return 0

    try:
        if args.show:
            plt.switch_backend("TkAgg")
        pending_figures = run(args.config, args.output_dir, show=args.show)
    except ImportError as error:
        print(f"Error: no interactive display available ({error})", file=sys.stderr)
        return 1
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.show:
        if not pending_figures:
            print("No figures to show.")
            return 0
        # Keep a reference so the button callbacks stay alive while the window is open.
        viewer = FigureViewer(pending_figures)
        plt.show()
    return 0
