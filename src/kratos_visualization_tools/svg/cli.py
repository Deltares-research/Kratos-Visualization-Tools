"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .geo_reader import parse_geo
from .msh_reader import parse_msh
from .mdpa_reader import parse_mdpa
from .render import geo_to_svg, msh_to_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GEO/MSH/MDPA file to styled SVG.")
    parser.add_argument("--input", required=True, help="Path to .geo, .msh or .mdpa file")
    parser.add_argument("--output", required=True, help="Path to .svg output")
    parser.add_argument("--width", type=int, default=2200, help="SVG width in px")
    parser.add_argument("--height", type=int, default=1200, help="SVG height in px")
    parser.add_argument("--padding", type=int, default=70, help="Padding in px")
    parser.add_argument(
        "--legend-config",
        help="Optional JSON file with legend entries and grouped names",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    legend_cfg: Optional[Dict[str, Any]] = None
    if args.legend_config:
        cfg_path = Path(args.legend_config)
        legend_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(legend_cfg, dict):
            raise ValueError("Legend config root must be a JSON object")
        legend_cfg = legend_cfg.get("legend") if "legend" in legend_cfg else legend_cfg
        if not isinstance(legend_cfg, dict):
            raise ValueError("Legend config must contain a JSON object")

    suffix = in_path.suffix.lower()

    if suffix in {".geo", ".mdpa"}:
        model = parse_mdpa(in_path) if suffix == ".mdpa" else parse_geo(in_path)
        geo_to_svg(
            model,
            out_path,
            width=args.width,
            height=args.height,
            padding=args.padding,
            legend_cfg=legend_cfg,
        )

        print(f"Wrote SVG: {out_path}")
        print(
            f"Parsed points={len(model.points)}, lines={len(model.lines)}, "
            f"splines={len(model.splines)}, loops={len(model.curve_loops)}"
        )
        print(
            f"Physical curves={len(model.physical_curves)}, "
            f"physical surfaces={len(model.physical_surfaces)}"
        )
    elif suffix == ".msh":
        msh_model = parse_msh(in_path)
        msh_to_svg(
            msh_model,
            out_path,
            width=args.width,
            height=args.height,
            padding=args.padding,
            legend_cfg=legend_cfg,
        )
        print(f"Wrote SVG: {out_path}")
        print(f"Parsed nodes={len(msh_model.nodes)}, elements={len(msh_model.elements)}")
    else:
        raise ValueError("Unsupported input extension. Use .geo, .msh or .mdpa")
