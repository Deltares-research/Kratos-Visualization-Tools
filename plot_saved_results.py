#!/usr/bin/env python3
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

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent / "src"))

from kratos_visualization_tools._compat import install_facade as _install_facade
from kratos_visualization_tools import saved_results as _implementation

_install_facade(_sys.modules[__name__], _implementation._compat_owners, _implementation._compat_modules)

if __name__ == "__main__":
    raise SystemExit(main())
