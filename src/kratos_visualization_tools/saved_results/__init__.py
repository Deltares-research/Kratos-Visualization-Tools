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
from .._compat import install_facade as _install_facade
from . import primitives as _primitives
from . import data as _data
from . import building as _building
from . import figures as _figures
from . import viewer as _viewer
from . import config as _config
from . import cli as _cli

_install_facade(
    _sys.modules[__name__],
    {
        "argparse": _cli,
        "csv": _data,
        "io": _primitives,
        "json": _building,
        "math": _primitives,
        "re": _config,
        "sys": _cli,
        "dataclass": _primitives,
        "field": _primitives,
        "Path": _primitives,
        "Any": _primitives,
        "Iterable": _primitives,
        "matplotlib": _primitives,
        "plt": _primitives,
        "Button": _viewer,
        "CONFIG_DOC": _config,
        "VALUE_SOURCES": _building,
        "DataSeries": _primitives,
        "AxisOptions": _primitives,
        "_plot_on_axis": _primitives,
        "_apply_axis_options": _primitives,
        "_unique_label_handles": _primitives,
        "PendingFigure": _primitives,
        "_pending_figures": _primitives,
        "_write": _primitives,
        "_render_preview": _primitives,
        "_save": _primitives,
        "make_single_plot": _primitives,
        "make_sub_plots": _primitives,
        "Mesh": _data,
        "read_mdpa": _data,
        "_looks_like_header": _data,
        "Table": _data,
        "read_table": _data,
        "_substitute": _building,
        "_transform": _building,
        "_coordinate": _building,
        "_nodal_value": _building,
        "_sorted_points": _building,
        "_series_style": _building,
        "SeriesBuilder": _building,
        "_axis_options": _figures,
        "_collect": _figures,
        "_min_max_annotation": _figures,
        "make_figure": _figures,
        "_emit_sub_plots": _figures,
        "FigureViewer": _viewer,
        "run": _config,
        "_expand_config": _config,
        "_natural_sort_key": _config,
        "_discover_stages": _config,
        "_make_figures": _config,
        "main": _cli,
    },
    (_primitives, _data, _building, _figures, _viewer, _config, _cli,),
)
