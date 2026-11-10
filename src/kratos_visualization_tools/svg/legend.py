"""Legend configuration parsing, styles, colours and group ownership."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .models import GeoModel
from .document import _hash_color


@dataclass
class LegendEntry:
    names: List[str]
    label: str
    style_key: str
    color: Optional[str] = None


@dataclass
class LegendSpec:
    curve_entries: List[LegendEntry]
    surface_entries: List[LegendEntry]
    curve_style_by_name: Dict[str, str]
    surface_style_by_name: Dict[str, str]
    max_curve_items: int
    max_surface_items: int
    phreatic_line_label: Optional[str] = None
    phreatic_line_points: Optional[List[Tuple[float, float]]] = None
    legend_top_left: Tuple[float, float] = (20.0, 20.0)
    show_rulers: bool = True

    def color(self, style_key: str) -> str:
        """Return the configured entry color for a style, else a hashed one."""
        for entry in self.curve_entries + self.surface_entries:
            if entry.style_key == style_key and entry.color:
                return entry.color
        return _hash_color(style_key)


def _parse_legend_entry(raw: Any, kind: str) -> Optional[Tuple[List[str], str]]:
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            return None
        return [name], name

    if not isinstance(raw, dict):
        return None

    if raw.get("show") is False:
        return None

    names: List[str] = []
    raw_names = raw.get("names")
    raw_name = raw.get("name")

    if isinstance(raw_names, list):
        names = [str(v).strip() for v in raw_names if str(v).strip()]
    elif isinstance(raw_name, str) and raw_name.strip():
        names = [raw_name.strip()]

    if not names:
        return None

    label = str(raw.get("label") or ", ".join(names)).strip()
    if not label:
        label = names[0]

    return names, label


def _configured_legend_color(raw: Any) -> Optional[str]:
    """Return a legend entry's optional ``#RRGGBB`` color in lower case."""
    if not isinstance(raw, dict) or "color" not in raw:
        return None
    color = raw["color"]
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError(f"Invalid legend color {color!r}; expected a #RRGGBB hex color")
    return color.lower()


def _legend_entries(
    names: List[str], raw_entries: Any, kind: str,
) -> Tuple[List[LegendEntry], Dict[str, str]]:
    styles = {name: name for name in names}
    if not isinstance(raw_entries, list):
        return [LegendEntry([name], name, name) for name in names], styles

    entries: List[LegendEntry] = []
    for idx, raw in enumerate(raw_entries):
        parsed = _parse_legend_entry(raw, kind)
        if not parsed:
            continue
        group_names, label = parsed
        existing = [name for name in group_names if name in styles]
        if not existing:
            continue
        style_key = f"{kind}_cfg_{idx}_{label}"
        styles.update(dict.fromkeys(existing, style_key))
        entries.append(LegendEntry(existing, label, style_key, _configured_legend_color(raw)))
    return entries, styles


def _unique_group_ids(
    groups: Dict[str, List[int]], entries: List[LegendEntry],
) -> Dict[str, List[int]]:
    """Assign each geometry id to one group so shared entities are drawn once.

    Groups listed in the legend config win in config order; the remaining
    groups follow from smallest to largest, so aggregate groups lose to
    their more specific parts.
    """
    configured = [name for entry in entries for name in entry.names if name in groups]
    configured_set = set(configured)
    remaining = sorted(
        (name for name in groups if name not in configured_set),
        key=lambda name: (len(groups[name]), name),
    )
    owner: Dict[int, str] = {}
    for name in dict.fromkeys(configured + remaining):
        for gid in groups[name]:
            owner.setdefault(abs(gid), name)

    unique: Dict[str, List[int]] = {}
    for name, ids in groups.items():
        seen: set = set()
        unique[name] = []
        for gid in ids:
            if owner[abs(gid)] == name and abs(gid) not in seen:
                seen.add(abs(gid))
                unique[name].append(gid)
    return unique


def _surface_regions(
    names: List[str], style_by_name: Dict[str, str],
) -> List[Tuple[str, List[str]]]:
    """Group surface names by legend style; each style is drawn as one region."""
    regions: Dict[str, List[str]] = {}
    for name in names:
        regions.setdefault(style_by_name.get(name, name), []).append(name)
    return list(regions.items())


def _build_legend_spec(model: GeoModel, legend_cfg: Optional[Dict[str, Any]]) -> LegendSpec:
    curve_names = sorted(model.physical_curves.keys())
    surface_names = sorted(model.physical_surfaces.keys())
    legend_cfg = legend_cfg or {}
    max_curve_items = int(legend_cfg.get("max_curves", 18))
    max_surface_items = int(legend_cfg.get("max_surfaces", 18))
    # "conditions" and "elements" are accepted as aliases of the GEO terms.
    curve_entries, curve_style_by_name = _legend_entries(
        curve_names, legend_cfg.get("curves", legend_cfg.get("conditions")), "curve"
    )
    surface_entries, surface_style_by_name = _legend_entries(
        surface_names, legend_cfg.get("surfaces", legend_cfg.get("elements")), "surface"
    )

    # Parse phreatic line config
    phreatic_line_label: Optional[str] = None
    phreatic_line_points: Optional[List[Tuple[float, float]]] = None
    
    phreatic_cfg = legend_cfg.get("phreatic_line")
    if isinstance(phreatic_cfg, dict):
        phreatic_line_label = str(phreatic_cfg.get("label", "Phreatic Line")).strip()
        raw_points = phreatic_cfg.get("points")
        if isinstance(raw_points, list) and len(raw_points) >= 2:
            try:
                phreatic_line_points = [(float(p[0]), float(p[1])) for p in raw_points if isinstance(p, (list, tuple)) and len(p) >= 2]
                if len(phreatic_line_points) < 2:
                    phreatic_line_points = None
            except (ValueError, TypeError):
                phreatic_line_points = None
    
    # Parse legend top-left anchor in SVG coordinates.
    legend_top_left: Tuple[float, float] = (20.0, 20.0)
    raw_top_left = legend_cfg.get("top_left")
    if isinstance(raw_top_left, (list, tuple)) and len(raw_top_left) >= 2:
        try:
            legend_top_left = (float(raw_top_left[0]), float(raw_top_left[1]))
        except (ValueError, TypeError):
            legend_top_left = (20.0, 20.0)
    elif isinstance(raw_top_left, dict):
        try:
            legend_top_left = (
                float(raw_top_left.get("x", 20.0)),
                float(raw_top_left.get("y", 20.0)),
            )
        except (ValueError, TypeError):
            legend_top_left = (20.0, 20.0)
    
    return LegendSpec(
        curve_entries=curve_entries,
        surface_entries=surface_entries,
        curve_style_by_name=curve_style_by_name,
        surface_style_by_name=surface_style_by_name,
        max_curve_items=max_curve_items,
        max_surface_items=max_surface_items,
        phreatic_line_label=phreatic_line_label,
        phreatic_line_points=phreatic_line_points,
        legend_top_left=legend_top_left,
        show_rulers=legend_cfg.get("rulers", True) is not False,
    )
