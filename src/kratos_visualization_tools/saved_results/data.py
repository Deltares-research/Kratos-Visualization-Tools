"""Saved MDPA coordinate and delimited-table readers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Mesh:
    coordinates: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    node_ids_per_sub_model_part: dict[str, list[int]] = field(default_factory=dict)

    def node_ids(self, sub_model_part_name: str) -> list[int]:
        # Accept both "PorousDomain.Sheet_Pile_Wall" and "Sheet_Pile_Wall"
        key = sub_model_part_name.split(".")[-1]
        if key not in self.node_ids_per_sub_model_part:
            raise KeyError(
                f"Sub model part '{sub_model_part_name}' not found in the mesh file"
            )
        return self.node_ids_per_sub_model_part[key]


def read_mdpa(mdpa_file_path: Path) -> Mesh:
    """Minimal .mdpa reader: nodal coordinates and sub model part node lists."""
    mesh = Mesh()
    section = None
    sub_model_part_stack: list[str] = []

    with open(mdpa_file_path, "r") as mdpa_file:
        for raw_line in mdpa_file:
            line = raw_line.split("//")[0].strip()
            if not line:
                continue

            if line.startswith("Begin Nodes"):
                section = "nodes"
                continue
            if line.startswith("Begin SubModelPartNodes"):
                section = "sub_model_part_nodes"
                continue
            if line.startswith("Begin SubModelPart "):
                sub_model_part_stack.append(line.split()[-1])
                mesh.node_ids_per_sub_model_part.setdefault(
                    sub_model_part_stack[-1], []
                )
                continue
            if line.startswith("End SubModelPartNodes"):
                section = None
                continue
            if line.startswith("End SubModelPart"):
                if sub_model_part_stack:
                    sub_model_part_stack.pop()
                continue
            if line.startswith("End "):
                section = None
                continue
            if line.startswith("Begin "):
                section = "other"
                continue

            if section == "nodes":
                words = line.replace(",", " ").split()
                mesh.coordinates[int(words[0])] = (
                    float(words[1]),
                    float(words[2]),
                    float(words[3]),
                )
            elif section == "sub_model_part_nodes" and sub_model_part_stack:
                mesh.node_ids_per_sub_model_part[sub_model_part_stack[-1]].append(
                    int(line.split()[0])
                )

    return mesh


def _looks_like_header(row: list[str]) -> bool:
    for cell in row:
        try:
            float(cell)
        except ValueError:
            return True
    return False


@dataclass
class Table:
    """A CSV/text table; rows are accessible by column name or by column index."""

    header: list[str] | None
    rows: list[list[str]]

    def column(self, key: int | str) -> list[str]:
        if isinstance(key, int):
            index = key
        else:
            if self.header is None:
                raise KeyError(
                    f"Column '{key}' requested, but the file has no header row"
                )
            if key not in self.header:
                raise KeyError(
                    f"Column '{key}' not found. Available columns: {self.header}"
                )
            index = self.header.index(key)
        return [row[index] for row in self.rows]

    def has_column(self, key: int | str) -> bool:
        try:
            self.column(key)
        except (KeyError, IndexError):
            return False
        return True


def read_table(file_path: Path, comment_prefix: str = "#") -> Table:
    """Read a comma- or whitespace-separated file, with or without a header."""
    with open(file_path, newline="", encoding="utf-8-sig") as data_file:
        lines = [
            line.strip().lstrip("\ufeff")
            for line in data_file
            if line.strip() and not line.strip().startswith(comment_prefix)
        ]

    if not lines:
        return Table(header=None, rows=[])

    delimiter = "," if "," in lines[0] else None
    if delimiter is None:
        split_rows = [line.split() for line in lines]
    else:
        split_rows = [row for row in csv.reader(lines, skipinitialspace=True)]
    split_rows = [[cell.strip() for cell in row] for row in split_rows]

    if _looks_like_header(split_rows[0]):
        return Table(header=split_rows[0], rows=split_rows[1:])
    return Table(header=None, rows=split_rows)
