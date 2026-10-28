"""GiD export filenames and JSON-driven line-to-CSV extraction."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np
from .calculations import finite_difference_gradient, project_element_to_nodes, tensor_component, vector_component
from .parsing import load_values, parse_post_msh, scan_result_headers
from .results import pick_step, resolve_mesh_result_paths


def build_output_path(prefix: str, step: str, location: Optional[str], suffix: str) -> Path:
    """Create a clean output filename from result metadata with sanitized separators."""
    safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix)).strip("_") or "result"
    safe_step = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(step)).replace(".", "_").strip("_") or "step"
    safe_location = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(location or "auto")).strip("_") or "auto"
    return Path(f"{safe_prefix}_{safe_step}_{safe_location}{suffix}")


def build_result_export_path(
    result_name: str,
    step: str,
    location: Optional[str],
    kind: str,
    component: Optional[str] = None,
    tensor_component: Optional[str] = None,
    extension: str = ".csv",
) -> Path:
    """Build a consistent export filename for profile and field outputs."""
    suffixes = [
        part for part in (
            component if component and component != "scalar" else None,
            tensor_component if tensor_component and tensor_component != "scalar" else None,
            kind,
        )
        if part is not None
    ]
    suffix = "_".join(suffixes)
    return build_output_path(result_name, step, location, f"_{suffix}{extension}")


def _parse_line_spec(spec: Dict[str, object]) -> Tuple[Tuple[float, float], Tuple[float, float], int]:
    line_obj = spec.get("line")
    start = end = None
    samples = None

    if isinstance(line_obj, dict):
        s = line_obj.get("start")
        e = line_obj.get("end")
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            start = (float(s[0]), float(s[1]))
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            end = (float(e[0]), float(e[1]))
        if "samples" in line_obj:
            samples = int(line_obj["samples"])

    if start is None or end is None:
        pts = spec.get("line_points")
        if isinstance(pts, (list, tuple)) and len(pts) >= 2:
            p0, p1 = pts[0], pts[1]
            if isinstance(p0, (list, tuple)) and len(p0) >= 2 and isinstance(p1, (list, tuple)) and len(p1) >= 2:
                start = (float(p0[0]), float(p0[1]))
                end = (float(p1[0]), float(p1[1]))

    if start is None or end is None:
        raise ValueError("JSON must define line coordinates as line.start/end or line_points.")

    if samples is None:
        if "line_samples" in spec:
            samples = int(spec["line_samples"])
        else:
            samples = 200

    if samples < 2:
        raise ValueError("Line sample count must be >= 2.")

    return start, end, samples


def _sample_line_values_idw(
    all_nodes: Dict[int, Tuple[float, float, float]],
    scalar_values: Dict[int, float],
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    nsamples: int,
) -> List[Tuple[float, float, float, float, float]]:
    node_ids = []
    xy = []
    vals = []
    for nid, v in scalar_values.items():
        if nid not in all_nodes or not np.isfinite(v):
            continue
        x, y, _ = all_nodes[nid]
        node_ids.append(nid)
        xy.append((x, y))
        vals.append(float(v))

    if not xy:
        return []

    xy_arr = np.array(xy, dtype=float)
    val_arr = np.array(vals, dtype=float)
    p0_arr = np.array(p0, dtype=float)
    p1_arr = np.array(p1, dtype=float)
    seg = p1_arr - p0_arr
    seg_len = float(np.linalg.norm(seg))

    out: List[Tuple[float, float, float, float, float]] = []
    eps = 1e-12

    for t in np.linspace(0.0, 1.0, nsamples):
        p = p0_arr + t * seg
        d2 = np.sum((xy_arr - p) ** 2, axis=1)

        nearest = int(np.argmin(d2))
        if d2[nearest] < 1e-16:
            v = float(val_arr[nearest])
        else:
            k = min(8, len(val_arr))
            idx = np.argpartition(d2, k - 1)[:k]
            w = 1.0 / (d2[idx] + eps)
            v = float(np.sum(w * val_arr[idx]) / np.sum(w))

        x = float(p[0])
        y = float(p[1])
        s = float(t * seg_len)
        out.append((t, s, x, y, v))

    return out


def export_line_csv_from_json(json_path: Path, csv_out: Optional[Path] = None) -> Path:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Input JSON must be an object.")

    cfg = raw.get("extract") if isinstance(raw.get("extract"), dict) else raw
    if not isinstance(cfg, dict):
        raise ValueError("Extraction config must be a JSON object.")

    file_bases = cfg.get("files")
    if not isinstance(file_bases, list) or not file_bases:
        raise ValueError("JSON must define a non-empty 'files' array with base names or file paths.")

    result_name = str(cfg.get("result", "WATER_PRESSURE"))
    step_req = str(cfg.get("step", "latest"))
    location_req = cfg.get("location")
    location_req = None if location_req in (None, "", "auto") else str(location_req)
    component = str(cfg.get("component", "magnitude"))

    p0, p1, nsamples = _parse_line_spec(cfg)

    if csv_out is not None:
        out_path = csv_out
    elif isinstance(cfg.get("output_csv"), str) and cfg.get("output_csv"):
        out_path = (json_path.parent / str(cfg.get("output_csv"))).resolve()
    else:
        out_path = (json_path.parent / f"{json_path.stem}_line_extract.csv").resolve()

    rows: List[Dict[str, object]] = []

    for base in file_bases:
        base_str = str(base)
        base_path = Path(base_str)
        if not base_path.is_absolute():
            base_path = (json_path.parent / base_path).resolve()

        msh_path, res_path = resolve_mesh_result_paths(str(base_path), base_dir=None)

        if not msh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {msh_path}")
        if not res_path.exists():
            raise FileNotFoundError(f"Result file not found: {res_path}")

        all_nodes, triangles, surface_elements, line_meshes, _ = parse_post_msh(msh_path)
        headers = scan_result_headers(res_path)
        if not headers:
            raise RuntimeError(f"No parsable Result blocks found in {res_path}")

        step, location = pick_step(headers, result_name, step_req, location_req)
        result_type, values = load_values(res_path, result_name, step, location)

        if location == "OnGaussPoints":
            line_elements = []
            for _, _, elems in line_meshes:
                line_elements.extend(elems)
            values = project_element_to_nodes(values, surface_elements, line_elements if line_elements else None)

        is_vector = result_type.lower() == "vector"
        scalar_values: Dict[int, float] = {}
        for nid, arr in values.items():
            if is_vector:
                scalar_values[nid] = vector_component(arr, component)
            else:
                scalar_values[nid] = float(arr[0]) if arr.size > 0 else float("nan")

        sampled = _sample_line_values_idw(all_nodes, scalar_values, p0, p1, nsamples)
        sample_s = np.array([s for _, s, _, _, _ in sampled], dtype=float)
        sample_values = np.array([v for _, _, _, _, v in sampled], dtype=float)
        sample_gradients = finite_difference_gradient(sample_s, sample_values)

        for i, (t, s, x, y, v) in enumerate(sampled):
            rows.append(
                {
                    "file_base": base_str,
                    "result": result_name,
                    "step": step,
                    "location": location,
                    "component": component if is_vector else "scalar",
                    "sample_index": i,
                    "t": t,
                    "s": s,
                    "x": x,
                    "y": y,
                    "value": v,
                    "gradient": float(sample_gradients[i]),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_base",
        "result",
        "step",
        "location",
        "component",
        "sample_index",
        "t",
        "s",
        "x",
        "y",
        "value",
        "gradient",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path
