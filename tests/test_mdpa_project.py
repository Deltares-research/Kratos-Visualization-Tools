"""
====================================================================

⚠️  AI-ASSISTED PROTOTYPE — PROVENANCE & VALIDATION NOTICE

====================================================================

Author: Gennady Markelov

This contribution (scripts and documentation) was generated with AI assistance and reviewed on selected internal test cases. It is provided as a prototype and should not be considered production-ready.
Please perform independent review, validation, and testing before use.
 
====================================================================
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_mdpa_project import (
    _normalize_submodelpart_name,
    _stage_from_process_label,
    parse_project_parameters,
    find_mdpa_from_project,
    visualize_mdpa,
)


class TestVisualizeMdpaLite(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_parse_project_parameters_extracts_stage_names_and_model_parts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = os.path.join(tmp_dir, "ProjectParameters.json")
            project_data = {
                "stages": {
                    "StageA": {
                        "stage_preprocess": {
                            "operations": [
                                {
                                    "name": "OperationA",
                                    "Parameters": {"model_part_name": "Soil"},
                                }
                            ]
                        },
                        "stage_settings": {
                            "solver_settings": {
                                "problem_domain_sub_model_part_list": ["Soil", "Wall"]
                            }
                        },
                    }
                },
                "processes": {
                    "ProcessGroup": [
                        {
                            "process_name": "ApplyPressure",
                            "Parameters": {"model_part_name": "Wall"},
                        }
                    ]
                },
            }
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f)

            mapping, stage_names, stage_referenced_submodels = parse_project_parameters(
                [project_path], base_dir=tmp_dir
            )

            self.assertEqual(["StageA"], stage_names)
            self.assertIn("Soil", mapping)
            self.assertIn("Wall", mapping)
            self.assertIn("StageA", stage_referenced_submodels)
            self.assertIn("Soil", stage_referenced_submodels["StageA"])
            self.assertIn("Wall", stage_referenced_submodels["StageA"])
            self.assertTrue(
                any("OperationA" in label for label in mapping["Soil"]),
                msg=f"Expected OperationA label in {mapping['Soil']}",
            )
            self.assertTrue(
                any("ApplyPressure" in label for label in mapping["Wall"]),
                msg=f"Expected ApplyPressure label in {mapping['Wall']}",
            )

    def test_find_mdpa_from_project_import_modeler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = os.path.join(tmp_dir, "ProjectParameters.json")
            mdpa_path = os.path.join(tmp_dir, "eemdijk.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write("Begin Nodes\n1 0 0 0\nEnd Nodes\n")
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stages": [
                            {
                                "stage_preprocess": {
                                    "modelers": [
                                        {
                                            "name": "Modelers.KratosMultiphysics.ImportMDPAModeler",
                                            "Parameters": {
                                                "input_filename": "eemdijk",
                                                "model_part_name": "PorousDomain",
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                    f,
                )

            self.assertEqual(mdpa_path, find_mdpa_from_project(project_path))

    def test_parse_project_parameters_normalizes_model_part_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = os.path.join(tmp_dir, "model_stages.json")
            project_data = {
                "stages": {
                    "StageB": {
                        "processes": [
                            {
                                "process_name": "Attach",
                                "Parameters": {"model_part_name": "Outer.ModelPart"},
                            }
                        ]
                    }
                }
            }
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f)

            mapping, stage_names, _ = parse_project_parameters(
                ["model_stages.json"], base_dir=tmp_dir
            )

            self.assertEqual(["StageB"], stage_names)
            self.assertIn("ModelPart", mapping)
            self.assertTrue(
                any("Attach" in label for label in mapping["ModelPart"]),
                msg=f"Expected Attach label in {mapping['ModelPart']}",
            )

    def test_parse_project_parameters_uses_standalone_filename_as_stage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = os.path.join(
                tmp_dir, "ProjectParameters_stage_construction.json"
            )
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "solver_settings": {
                            "problem_domain_sub_model_part_list": ["Clay_cap", "Clay"],
                            "body_domain_sub_model_part_list": ["Clay_cap", "Clay", "Sand"],
                            "processes_sub_model_part_list": [
                                "Clay_cap", "Clay", "Sand", "River_load"
                            ],
                        },
                        "processes": [
                            {
                                "process_name": "ApplyPressure",
                                "Parameters": {"model_part_name": "Soil"},
                            }
                        ]
                    },
                    f,
                )

            mapping, stage_names, references = parse_project_parameters(
                [project_path], base_dir=tmp_dir
            )

            self.assertEqual(["stage_construction"], stage_names)
            self.assertEqual(
                ["[stage_construction] ApplyPressure"], mapping["Soil"]
            )
            self.assertTrue(
                any("Used as problem_domain" in label for label in mapping["Clay"])
            )
            self.assertTrue(
                any("Used as body_domain" in label for label in mapping["Sand"])
            )
            self.assertTrue(
                any("Used by processes" in label for label in mapping["River_load"])
            )
            self.assertEqual(
                {"Soil", "Clay_cap", "Clay", "Sand", "River_load"},
                references["stage_construction"],
            )

    def test_parse_project_parameters_orders_orchestrator_and_standalone_stages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            standalone_paths = []
            for stage_name in ("stage10", "stage2", "stage1"):
                path = os.path.join(
                    tmp_dir, f"ProjectParameters_{stage_name}.json"
                )
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"solver_settings": {}}, f)
                standalone_paths.append(path)

            _, standalone_stages, _ = parse_project_parameters(
                standalone_paths, base_dir=tmp_dir
            )

            self.assertEqual(
                ["stage1", "stage2", "stage10"], standalone_stages
            )

            orchestrator_path = os.path.join(tmp_dir, "orchestrator.json")
            with open(orchestrator_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"stages": {"stage10": {}, "stage1": {}, "stage2": {}}},
                    f,
                )

            _, orchestrator_stages, _ = parse_project_parameters(
                [orchestrator_path], base_dir=tmp_dir
            )

            self.assertEqual(
                ["stage10", "stage1", "stage2"], orchestrator_stages
            )

    def test_parse_project_parameters_supports_stage_lists_and_extended_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_path = os.path.join(tmp_dir, "stage_list.json")
            project_data = {
                "stages": [
                    {
                        "name": "Construction",
                        "stage_preprocess": {
                            "modelers": [
                                {
                                    "name": "BuildModel",
                                    "Parameters": {
                                        "model_part_name_list": ["Domain.Soil"]
                                    },
                                }
                            ]
                        },
                        "stage_settings": {
                            "solver_settings": {
                                "problem_domain_sub_model_part_list": ["Domain.Soil"],
                                "body_domain_sub_model_part_list": [
                                    "Domain.Soil",
                                    "Domain.Wall",
                                ],
                            }
                        },
                    },
                    {
                        "stage_name": "Loading",
                        "processes": [
                            {
                                "python_module": "assign_process",
                                "Parameters": {
                                    "master_model_part_name": "Domain.Wall",
                                    "slave_model_part_name": "Domain.Interface",
                                    "variable_names": ["DISPLACEMENT", 7],
                                },
                            }
                        ],
                    },
                    "not-a-stage-dict",
                ]
            }
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f)

            mapping, stage_names, stage_referenced_submodels = parse_project_parameters(
                [project_path], base_dir=tmp_dir
            )

            self.assertEqual(["Construction", "Loading", "stage_3"], stage_names)
            self.assertTrue(any("BuildModel" in label for label in mapping["Soil"]))
            self.assertTrue(
                any("Used as problem_domain" in label for label in mapping["Soil"])
            )
            self.assertTrue(
                any("Used as body_domain" in label for label in mapping["Wall"])
            )
            self.assertEqual(
                ["[Loading] assign_process (DISPLACEMENT)"],
                mapping["Wall"][-1:],
            )
            self.assertIn("Interface", mapping)
            self.assertIn("Soil", stage_referenced_submodels["Construction"])
            self.assertIn("Wall", stage_referenced_submodels["Loading"])

    def test_parse_project_parameters_reports_missing_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = os.path.join(tmp_dir, "invalid.json")
            with open(invalid_path, "w", encoding="utf-8") as f:
                f.write("{invalid json")

            with patch("builtins.print") as print_mock:
                mapping, stage_names, references = parse_project_parameters(
                    ["missing.json", invalid_path], base_dir=tmp_dir
                )

            self.assertEqual({}, dict(mapping))
            self.assertEqual([], stage_names)
            self.assertEqual({}, dict(references))
            printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
            self.assertIn("Could not find project file 'missing.json'", printed)
            self.assertIn("Could not parse project file", printed)

    def test_helpers_for_submodel_and_stage_labels(self):
        self.assertEqual("Wall", _normalize_submodelpart_name("Group.Wall"))
        self.assertEqual("Wall", _normalize_submodelpart_name("Wall"))
        self.assertEqual("Global", _stage_from_process_label("ApplyPressure"))
        self.assertEqual("StageA", _stage_from_process_label("[StageA] ApplyPressure"))

    def test_visualize_mdpa_auto_detects_stage_files_in_mdpa_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "demo_model.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
    2
  End SubModelPartNodes
End SubModelPart

Begin SubModelPart Wall
  Begin SubModelPartNodes
    2
    3
  End SubModelPartNodes
End SubModelPart
"""
                )

            with open(os.path.join(tmp_dir, "ProjectParameters.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stages": {
                            "StageA": {
                                "processes": [
                                    {
                                        "process_name": "StageAProcess",
                                        "Parameters": {"model_part_name": "Soil"},
                                    }
                                ]
                            }
                        }
                    },
                    f,
                )

            with open(os.path.join(tmp_dir, "extra_stage_dump.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stages": {
                            "StageB": {
                                "processes": [
                                    {
                                        "process_name": "StageBProcess",
                                        "Parameters": {"model_part_name": "Wall"},
                                    }
                                ]
                            }
                        }
                    },
                    f,
                )

            with patch("matplotlib.pyplot.show"):
                visualize_mdpa(mdpa_path, [])

            fig = plt.gcf()
            self.assertTrue(fig is not None)
            ax = next(
                axes
                for axes in fig.axes
                if axes.get_title().startswith("MDPA Visualization")
            )
            self.assertTrue(len(ax.collections) >= 0)

    def test_visualize_mdpa_recovers_stage_names_from_stage_blocks_and_resolves_project_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "project_model.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
    2

  End SubModelPartNodes
End SubModelPart

Begin SubModelPart Wall
  Begin SubModelPartNodes
    2
    3
  End SubModelPartNodes
End SubModelPart
"""
                )

            stage_json = os.path.join(tmp_dir, "stages_fallback.json")
            with open(stage_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stages": [
                            {"name": "Construction"},
                            {"stage_name": "Loading"},
                            "not-a-stage-dict",
                        ]
                    },
                    f,
                )

            with patch("matplotlib.pyplot.show"):
                visualize_mdpa(mdpa_path, ["stages_fallback.json"])

            fig = plt.gcf()
            ax = next(
                axes
                for axes in fig.axes
                if axes.get_title().startswith("MDPA Visualization")
            )
            self.assertTrue(ax is not None)

            stage_title = ax.get_title()
            self.assertIn("MDPA Visualization", stage_title)

            other_json = os.path.join(tmp_dir, "ProjectParameters.json")
            with open(other_json, "w", encoding="utf-8") as f:
                json.dump({"solver_settings": {"problem_domain_sub_model_part_list": ["Soil"]}}, f)

            with patch("matplotlib.pyplot.show"):
                visualize_mdpa(mdpa_path, ["."])

            self.assertTrue(plt.gcf() is not None)

    def test_visualize_mdpa_recovers_stages_when_parser_returns_no_stage_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "recovery_model.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
  End SubModelPartNodes
End SubModelPart
"""
                )

            with open(os.path.join(tmp_dir, "stage_dict.json"), "w", encoding="utf-8") as f:
                json.dump({"stages": {"DictStage": {}}}, f)
            with open(os.path.join(tmp_dir, "stage_list.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stages": [
                            {"name": "NamedStage"},
                            {"stage_name": "StageName"},
                            {"id": "IdStage"},
                            {},
                            "not-a-stage-dict",
                        ]
                    },
                    f,
                )
            with open(os.path.join(tmp_dir, "invalid_stage.json"), "w", encoding="utf-8") as f:
                f.write("{invalid json")

            project_paths = [
                "stage_dict.json",
                "stage_list.json",
                "missing_stage.json",
                "invalid_stage.json",
            ]
            with patch(
                "visualize_mdpa_project.parse_project_parameters",
                return_value=({}, [], {}),
            ), patch("matplotlib.pyplot.show"), patch("builtins.print") as print_mock:
                visualize_mdpa(mdpa_path, project_paths)

            printed = "\n".join(
                str(call.args[0]) for call in print_mock.call_args_list if call.args
            )
            self.assertIn(
                "Recovered stage names from JSON 'stages' blocks: DictStage, NamedStage, StageName, IdStage, stage_4, stage_5",
                printed,
            )


if __name__ == "__main__":
    unittest.main()
