#!/usr/bin/env python3
"""
====================================================================

⚠️  AI-ASSISTED PROTOTYPE — PROVENANCE & VALIDATION NOTICE

====================================================================

Author: Gennady Markelov

This contribution (scripts and documentation) was generated with AI assistance and reviewed on selected internal test cases. It is provided as a prototype and should not be considered production-ready.
Please perform independent review, validation, and testing before use.
 
====================================================================

Lightweight MDPA visualizer using matplotlib (no heavy VTK dependencies).

Features:
- Parse Kratos `.mdpa` file for Nodes and SubModelPart definitions
- Visualize nodes colored by SubModelPart membership in 2D scatter plot
- Click on nodes to see SubModelPart info and associated processes
- Much lighter than pyvista version (matplotlib only)

Usage:
    python visualize_mdpa_project.py path/to/model.mdpa [--project params.json]
"""
import argparse
import colorsys
import json
import os
import re
import subprocess
import sys
import textwrap
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from kratos_visualization_tools.mdpa import (
    _connection_paths,
    _normal_segments,
    _normal_vector,
    _normalize_submodelpart_name,
    _stage_from_process_label,
    ParsedMdpa,
    parse_mdpa,
    parse_mdpa_data,
)
from kratos_visualization_tools.project import (
    find_mdpa_from_project,
    parse_project_parameters,
)
from kratos_visualization_tools import visualization as _visualization
from kratos_visualization_tools.cli import main


def visualize_mdpa(mdpa_path, project_paths):
    """Compatibility wrapper retaining legacy module-level monkeypatch points."""
    for name in (
        "Button", "CheckButtons", "Slider", "TextBox", "parse_project_parameters",
        "plt", "np", "Path", "colorsys", "json", "os", "re", "subprocess",
        "sys", "textwrap", "warnings", "datetime", "perf_counter",
        "_normalize_submodelpart_name", "_stage_from_process_label",
    ):
        setattr(_visualization, name, globals()[name])
    return _visualization.visualize_mdpa(mdpa_path, project_paths)


if __name__ == "__main__":
    main()
