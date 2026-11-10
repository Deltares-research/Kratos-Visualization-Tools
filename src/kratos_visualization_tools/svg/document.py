"""SVG namespace, document skeleton, fill patterns and stable colours."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def _hash_color(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    r = 40 + (r % 170)
    g = 40 + (g % 170)
    b = 40 + (b % 170)
    return f"#{r:02x}{g:02x}{b:02x}"


def _pick_pattern(name: str) -> str:
    patterns = ["pat_diag", "pat_cross", "pat_dots"]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    idx = int(digest[-2:], 16) % len(patterns)
    return patterns[idx]


def _pattern_defs(svg: ET.Element) -> None:
    defs = ET.SubElement(svg, f"{{{SVG_NS}}}defs")

    p1 = ET.SubElement(defs, f"{{{SVG_NS}}}pattern", {
        "id": "pat_diag", "patternUnits": "userSpaceOnUse", "width": "8", "height": "8"
    })
    ET.SubElement(p1, f"{{{SVG_NS}}}path", {
        "d": "M 0 8 L 8 0", "stroke": "#444", "stroke-width": "1", "fill": "none"
    })

    p2 = ET.SubElement(defs, f"{{{SVG_NS}}}pattern", {
        "id": "pat_cross", "patternUnits": "userSpaceOnUse", "width": "8", "height": "8"
    })
    ET.SubElement(p2, f"{{{SVG_NS}}}path", {
        "d": "M 0 8 L 8 0 M 0 0 L 8 8", "stroke": "#444", "stroke-width": "1", "fill": "none"
    })

    p3 = ET.SubElement(defs, f"{{{SVG_NS}}}pattern", {
        "id": "pat_dots", "patternUnits": "userSpaceOnUse", "width": "8", "height": "8"
    })
    ET.SubElement(p3, f"{{{SVG_NS}}}circle", {
        "cx": "2", "cy": "2", "r": "1.2", "fill": "#444"
    })


def _new_svg(width: int, height: int) -> ET.Element:
    svg = ET.Element(
        f"{{{SVG_NS}}}svg",
        {"width": str(width), "height": str(height), "viewBox": f"0 0 {width} {height}"},
    )
    ET.SubElement(svg, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "#ffffff"
    })
    return svg
