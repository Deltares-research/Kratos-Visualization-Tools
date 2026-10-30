import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kratos_visualization_tools as tools
from kratos_visualization_tools import cli, mdpa, project, visualization
import visualize_mdpa_project as legacy


class TestMdpaEntryPoints(unittest.TestCase):
    def test_legacy_exports_share_package_implementations(self):
        for module, names in (
            (mdpa, (
                "GeometryType", "ENTITY_2D3N_GEOMETRIES", "ParsedMdpa",
                "parse_mdpa", "parse_mdpa_data", "_build_face_separation",
                "_separated_coordinates", "_resolve_constraint_relations",
            )),
            (project, (
                "parse_project_parameters", "find_mdpa_from_project",
                "_resolve_stage_activity",
            )),
            (visualization, ("visualize_mdpa",)),
            (cli, ("main",)),
        ):
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(getattr(legacy, name), getattr(module, name))
        self.assertIs(tools.visualize_mdpa, visualization.visualize_mdpa)

    def test_legacy_patches_propagate_and_restore(self):
        original_parser = project.parse_project_parameters
        with patch.object(legacy, "parse_project_parameters") as parser:
            self.assertIs(parser, project.parse_project_parameters)
            self.assertIs(parser, visualization.parse_project_parameters)
        self.assertIs(original_parser, project.parse_project_parameters)
        self.assertIs(original_parser, visualization.parse_project_parameters)

        original_separation = mdpa._build_face_separation
        with patch.object(legacy, "_build_face_separation") as separation:
            self.assertIs(separation, mdpa._build_face_separation)
            self.assertIs(separation, visualization._build_face_separation)
        self.assertIs(original_separation, mdpa._build_face_separation)
        self.assertIs(original_separation, visualization._build_face_separation)

    def test_console_entry_point_uses_legacy_patch_and_forwards_arguments(self):
        with patch.object(legacy, "visualize_mdpa") as viewer:
            with patch.object(sys, "argv", [
                "kratos-visualize-mdpa", "model.mdpa", "-p", "ProjectParameters.json",
            ]):
                cli.main()
            viewer.assert_called_once_with("model.mdpa", ["ProjectParameters.json"])

    def test_project_only_entry_point_resolves_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.mdpa"
            model_path.write_text("Begin Nodes\n1 0 0 0\nEnd Nodes\n", encoding="utf-8")
            project_path = Path(directory) / "ProjectParameters.json"
            project_path.write_text(json.dumps({
                "modelers": [{
                    "name": "Modelers.KratosMultiphysics.ImportMDPAModeler",
                    "Parameters": {"input_filename": "model"},
                }],
            }), encoding="utf-8")
            with patch.object(legacy, "visualize_mdpa") as viewer:
                with patch.object(sys, "argv", [
                    "kratos-visualize-mdpa", "-p", str(project_path),
                ]):
                    cli.main()
                viewer.assert_called_once_with(str(model_path), [str(project_path)])
