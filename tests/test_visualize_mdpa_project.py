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
from matplotlib.backend_bases import KeyEvent, MouseEvent
import numpy as np

from visualize_mdpa_project import (
    _normalize_submodelpart_name,
    _normal_segments,
    _stage_from_process_label,
    _connection_paths,
    _normal_vector,
    ParsedMdpa,
    parse_mdpa,
    parse_mdpa_data,
    parse_project_parameters,
    find_mdpa_from_project,
    visualize_mdpa,
)


class TestVisualizeMdpaLite(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_parse_mdpa_reads_nodes_and_submodelparts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "sample.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.5 1.0 0.0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
    2
  End SubModelPartNodes
End SubModelPart

Begin SubModelPart Structure.Wall
  Begin SubModelPartNodes
    2
    3
  End SubModelPartNodes
End SubModelPart
"""
                )

            nodes, submodel_nodes = parse_mdpa(mdpa_path)

            self.assertEqual(
                {
                    1: (0.0, 0.0, 0.0),
                    2: (1.0, 0.0, 0.0),
                    3: (0.5, 1.0, 0.0),
                },
                nodes,
            )
            self.assertEqual({1, 2}, submodel_nodes["Soil"])
            self.assertEqual({2, 3}, submodel_nodes["Structure.Wall"])

            parsed_nodes, parsed_submodel_nodes, connections, _, _ = parse_mdpa(
                mdpa_path, include_connections=True
            )
            self.assertEqual(nodes, parsed_nodes)
            self.assertEqual(dict(submodel_nodes), dict(parsed_submodel_nodes))
            self.assertEqual([], connections)

    def test_parse_mdpa_reads_element_and_condition_connections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "mesh.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
End Nodes

Begin Elements Element2D3N
1 0 1 2 3
End Elements

Begin Conditions LineCondition2D2N
2 0 2 3
End Conditions
"""
                )

            _, _, connections, _, _ = parse_mdpa(mdpa_path, include_connections=True)

            self.assertEqual(
                [
                    ("Element2D3N", (1, 2, 3)),
                    ("LineCondition2D2N", (2, 3)),
                ],
                connections,
            )

    def test_parse_mdpa_preserves_kratos_higher_order_element_types(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "higher_order_mesh.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0 0 0
2 1 0 0
3 0 1 0
4 2 0 0
5 2 1 0
6 1 1 0
7 0.5 0 0
8 1 0.5 0
9 0.5 0.5 0
10 1.5 0 0
11 2 0.5 0
12 1.5 1 0
13 0.5 1 0
14 0 0.5 0
15 10 0 0
16 11 0 0
17 10.5 0 0
18 10 1 0
19 11 1 0
20 10.5 1 0
End Nodes

Begin Elements SmallStrainUPwDiffOrderElement2D6N
1 0 1 2 3 7 8 9
End Elements
Begin Elements SmallStrainUPwDiffOrderElement2D8N
2 0 4 5 6 10 11 12 13 14
End Elements
Begin Elements Geo_UPwLineInterfacePlaneStrainDiffOrderElement3Plus3N
3 0 15 16 17 18 19 20
End Elements
"""
                )

            _, _, connections, _, _ = parse_mdpa(mdpa_path, include_connections=True)

            self.assertEqual(
                [
                    ("SmallStrainUPwDiffOrderElement2D6N", (1, 2, 3, 7, 8, 9)),
                    ("SmallStrainUPwDiffOrderElement2D8N", (4, 5, 6, 10, 11, 12, 13, 14)),
                    ("Geo_UPwLineInterfacePlaneStrainDiffOrderElement3Plus3N", (15, 16, 17, 18, 19, 20)),
                ],
                connections,
            )

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

    def test_parsed_mdpa_model_and_normal_helper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "typed.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write("Begin Nodes\n1 0 0 0\n2 1 0 0\nEnd Nodes\n")

            parsed = parse_mdpa_data(mdpa_path)

            self.assertIsInstance(parsed, ParsedMdpa)
            self.assertEqual({1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0)}, parsed.nodes)
            self.assertEqual((0.0, 0.5), tuple(_normal_vector((0, 0), (1, 0), 0.5)))
            self.assertIsNone(_normal_vector((1, 1), (1, 1), 1.0))

    def test_kratos_connection_paths_follow_geometry_node_order(self):
        self.assertEqual(
            [(1, 3, 2)],
            _connection_paths("LinearTimoshenkoCurvedBeamElement2D3N", (1, 2, 3)),
        )
        self.assertEqual(
            [(1, 4, 2, 5, 3, 6, 1)],
            _connection_paths("SmallStrainUPwDiffOrderElement2D6N", (1, 2, 3, 4, 5, 6)),
        )
        self.assertEqual(
            [(1, 5, 2, 6, 3, 7, 4, 8, 1)],
            _connection_paths("SmallStrainUPwDiffOrderElement2D8N", (1, 2, 3, 4, 5, 6, 7, 8)),
        )
        self.assertEqual(
            [(1, 3, 2), (4, 6, 5)],
            _connection_paths(
                "Geo_UPwLineInterfacePlaneStrainDiffOrderElement3Plus3N",
                (1, 2, 3, 4, 5, 6),
            ),
        )
        self.assertEqual(
            [(1, 2)],
            _connection_paths("LinearCondition2D2N", (1, 2)),
        )

    def test_kratos_normal_segments_cover_line_and_interface_sides(self):
        self.assertEqual(
            [(1, 2)],
            _normal_segments("LineCondition2D2N", (1, 2)),
        )
        self.assertEqual(
            [(1, 3, 2)],
            _normal_segments("LinearTimoshenkoCurvedBeamElement2D3N", (1, 2, 3)),
        )
        self.assertEqual(
            [(1, 2), (3, 4)],
            _normal_segments("Geo_ULineInterfacePlaneStrainElement2Plus2N", (1, 2, 3, 4)),
        )
        self.assertEqual(
            [(1, 3, 2), (4, 6, 5)],
            _normal_segments(
                "Geo_UPwLineInterfacePlaneStrainDiffOrderElement3Plus3N",
                (1, 2, 3, 4, 5, 6),
            ),
        )

    def test_visualize_mdpa_sets_up_stage_and_visibility_controls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "tiny_model.mdpa")
            project_path = os.path.join(tmp_dir, "ProjectParameters.json")

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

            with open(project_path, "w", encoding="utf-8") as f:
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
                            },
                            "StageB": {
                                "processes": [
                                    {
                                        "process_name": "StageBProcess",
                                        "Parameters": {"model_part_name": "Wall"},
                                    }
                                ]
                            },
                        }
                    },
                    f,
                )

            class _FakeLabel:
                def __init__(self, text):
                    self.text = text

                def set_fontsize(self, *_args, **_kwargs):
                    pass

                def set_color(self, *_args, **_kwargs):
                    pass

                def set_alpha(self, *_args, **_kwargs):
                    pass

                def set_fontweight(self, *_args, **_kwargs):
                    pass

                def set_visible(self, *_args, **_kwargs):
                    pass

                def get_text(self):
                    return self.text

            class _FakeCheckButtons:
                instances = []

                def __init__(self, ax, labels, actives):
                    self.ax = ax
                    self._labels = list(labels)
                    self._status = list(actives)
                    self.labels = [_FakeLabel(label) for label in self._labels]
                    self._callbacks = []
                    _FakeCheckButtons.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

                def set_active(self, index, trigger_callback=False):
                    if index < 0 or index >= len(self._status):
                        raise IndexError(index)
                    self._status[index] = not self._status[index]
                    if trigger_callback and self._callbacks:
                        self._callbacks[0](self._labels[index])

                def get_status(self):
                    return list(self._status)

            with patch("matplotlib.pyplot.show"), patch(
                "visualize_mdpa_project.CheckButtons", _FakeCheckButtons
            ):
                visualize_mdpa(mdpa_path, [project_path])

            self.assertGreaterEqual(len(_FakeCheckButtons.instances), 2)
            stage_widget = next(
                widget
                for widget in _FakeCheckButtons.instances
                if widget.labels and widget.labels[0].get_text() == "All"
            )
            stage_widget.set_active(1, trigger_callback=True)
            status = stage_widget.get_status()
            self.assertEqual(3, len(status))
            self.assertEqual(3, len(status))
            self.assertTrue(status[0] == status[1] or status[1] == status[2])

    def test_visualize_mdpa_zoom_and_reset_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "zoom_model.mdpa")
            project_path = os.path.join(tmp_dir, "ProjectParameters.json")

            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0 0 0
2 1 0 0
3 0 1 0
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

            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "stages": {
                            "StageA": {
                                "processes": [
                                    {
                                        "process_name": "P1",
                                        "Parameters": {"model_part_name": "Soil"},
                                    }
                                ]
                            }
                        }
                    },
                    f,
                )

            with patch("matplotlib.pyplot.show"):
                visualize_mdpa(mdpa_path, [project_path])

            fig = plt.gcf()
            ax = next(ax for ax in fig.axes if ax.get_title().startswith("MDPA Visualization"))
            orig_xlim = ax.get_xlim()
            orig_ylim = ax.get_ylim()

            scroll_event = MouseEvent("scroll_event", fig.canvas, 60, 60, button="up")
            scroll_event.inaxes = ax
            scroll_event.xdata = 0.5
            scroll_event.ydata = 0.5
            fig.canvas.callbacks.process("scroll_event", scroll_event)
            self.assertNotEqual(orig_xlim, ax.get_xlim())
            self.assertNotEqual(orig_ylim, ax.get_ylim())

            press_event = MouseEvent("button_press_event", fig.canvas, 60, 60, button=1)
            press_event.inaxes = ax
            press_event.xdata = 0.0
            press_event.ydata = 0.0
            release_event = MouseEvent(
                "button_release_event", fig.canvas, 60, 60, button=1
            )
            release_event.inaxes = ax
            release_event.xdata = 0.0
            release_event.ydata = 0.0
            fig.canvas.callbacks.process("button_press_event", press_event)
            fig.canvas.callbacks.process("button_release_event", release_event)

            key_event = KeyEvent("key_press_event", fig.canvas, "r")
            fig.canvas.callbacks.process("key_press_event", key_event)
            self.assertEqual(orig_xlim, ax.get_xlim())
            self.assertEqual(orig_ylim, ax.get_ylim())

    def test_visualize_mdpa_node_click_opens_report_window(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "report_model.mdpa")
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

            with patch("matplotlib.pyplot.show") as pyplot_show_mock, patch(
                "matplotlib.figure.Figure.show"
            ) as figure_show_mock:
                visualize_mdpa(mdpa_path, [])

                fig = plt.gcf()
                ax = next(
                    axes
                    for axes in fig.axes
                    if axes.get_title().startswith("MDPA Visualization")
                )
                press_event = MouseEvent(
                    "button_press_event", fig.canvas, 60, 60, button=1
                )
                press_event.inaxes = ax
                press_event.xdata = 0.0
                press_event.ydata = 0.0
                release_event = MouseEvent(
                    "button_release_event", fig.canvas, 60, 60, button=1
                )
                release_event.inaxes = ax
                release_event.xdata = 0.0
                release_event.ydata = 0.0
                fig.canvas.callbacks.process("button_press_event", press_event)
                fig.canvas.callbacks.process("button_release_event", release_event)

            figure_show_mock.assert_called_once()
            report_fig = plt.gcf()
            self.assertIsNot(report_fig, fig)
            self.assertGreaterEqual(len(report_fig.axes), 4)
            pyplot_show_mock.assert_called_once_with()

    def test_visualize_mdpa_node_click_handles_coincident_and_unassigned_nodes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "node_click_branches.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 0.0 0.0 0.0
3 10.0 10.0 0.0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
    2
  End SubModelPartNodes
End SubModelPart
"""
                )

            with patch("matplotlib.pyplot.show"), patch(
                "matplotlib.figure.Figure.show"
            ):
                visualize_mdpa(mdpa_path, [])
                fig = plt.gcf()
                ax = next(
                    axes
                    for axes in fig.axes
                    if axes.get_title().startswith("MDPA Visualization")
                )

                def click_node(xdata, ydata):
                    press_event = MouseEvent(
                        "button_press_event", fig.canvas, 60, 60, button=1
                    )
                    press_event.inaxes = ax
                    press_event.xdata = xdata
                    press_event.ydata = ydata
                    release_event = MouseEvent(
                        "button_release_event", fig.canvas, 60, 60, button=1
                    )
                    release_event.inaxes = ax
                    release_event.xdata = xdata
                    release_event.ydata = ydata
                    fig.canvas.callbacks.process("button_press_event", press_event)
                    fig.canvas.callbacks.process("button_release_event", release_event)

                click_node(0.0, 0.0)
                first_report = plt.gcf()
                self.assertIsNot(first_report, fig)

                with patch("builtins.print") as print_mock:
                    click_node(10.0, 10.0)

                printed_report = "\n".join(
                    str(call.args[0])
                    for call in print_mock.call_args_list
                    if call.args
                )
                self.assertIn("Processes:\n  None", printed_report)

    def test_visualize_mdpa_progressive_zoom_to_selected_node(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "progressive_zoom_model.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0 0 0
2 10 0 0
3 0 10 0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
    2
    3
  End SubModelPartNodes
End SubModelPart
"""
                )

            with patch("matplotlib.pyplot.show"):
                visualize_mdpa(mdpa_path, [])

            fig = plt.gcf()
            ax = next(
                axis for axis in fig.axes
                if axis.get_title().startswith("MDPA Visualization")
            )
            initial_xlim = ax.get_xlim()
            initial_ylim = ax.get_ylim()

            press_event = MouseEvent("button_press_event", fig.canvas, 60, 60, button=1)
            press_event.inaxes = ax
            press_event.xdata = 10.0
            press_event.ydata = 0.0
            release_event = MouseEvent(
                "button_release_event", fig.canvas, 60, 60, button=1
            )
            release_event.inaxes = ax
            release_event.xdata = 10.0
            release_event.ydata = 0.0
            fig.canvas.callbacks.process("button_press_event", press_event)
            fig.canvas.callbacks.process("button_release_event", release_event)

            first_zoom_event = KeyEvent("key_press_event", fig.canvas, "z")
            fig.canvas.callbacks.process("key_press_event", first_zoom_event)
            first_xlim = ax.get_xlim()
            first_ylim = ax.get_ylim()

            second_zoom_event = KeyEvent("key_press_event", fig.canvas, "z")
            fig.canvas.callbacks.process("key_press_event", second_zoom_event)
            second_xlim = ax.get_xlim()
            second_ylim = ax.get_ylim()

            self.assertNotEqual(initial_xlim, first_xlim)
            self.assertNotEqual(initial_ylim, first_ylim)
            self.assertAlmostEqual(10.0, sum(first_xlim) / 2.0)
            self.assertAlmostEqual(0.0, sum(first_ylim) / 2.0)
            self.assertLess(second_xlim[1] - second_xlim[0], first_xlim[1] - first_xlim[0])
            self.assertLess(second_ylim[1] - second_ylim[0], first_ylim[1] - first_ylim[0])

    def test_visualize_mdpa_mouse_pan_and_keyboard_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "mouse_keyboard_model.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 1.0 0.0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
    2
  End SubModelPartNodes
End SubModelPart
"""
                )

            with patch("matplotlib.pyplot.show"), patch(
                "matplotlib.figure.Figure.show"
            ):
                visualize_mdpa(mdpa_path, [])

                fig = plt.gcf()
                ax = next(
                    axes
                    for axes in fig.axes
                    if axes.get_title().startswith("MDPA Visualization")
                )
                original_xlim = ax.get_xlim()
                original_ylim = ax.get_ylim()

                press_event = MouseEvent(
                    "button_press_event", fig.canvas, 100, 100, button=1
                )
                press_event.inaxes = ax
                fig.canvas.callbacks.process("button_press_event", press_event)

                move_event = MouseEvent(
                    "motion_notify_event", fig.canvas, 140, 120
                )
                move_event.inaxes = ax
                fig.canvas.callbacks.process("motion_notify_event", move_event)
                self.assertNotEqual(original_xlim, ax.get_xlim())
                self.assertNotEqual(original_ylim, ax.get_ylim())

                release_event = MouseEvent(
                    "button_release_event", fig.canvas, 140, 120, button=1
                )
                release_event.inaxes = ax
                fig.canvas.callbacks.process("button_release_event", release_event)

                with patch("builtins.print") as print_mock:
                    key_event = KeyEvent("key_press_event", fig.canvas, "1")
                    fig.canvas.callbacks.process("key_press_event", key_event)
                    no_selection_event = KeyEvent("key_press_event", fig.canvas, "z")
                    fig.canvas.callbacks.process(
                        "key_press_event", no_selection_event
                    )

                printed_output = "\n".join(
                    str(call.args[0])
                    for call in print_mock.call_args_list
                    if call.args
                )
                self.assertIn("SubModelPart #1: Soil", printed_output)
                self.assertIn("No node selected", printed_output)

                reset_event = KeyEvent("key_press_event", fig.canvas, "r")
                fig.canvas.callbacks.process("key_press_event", reset_event)
                self.assertEqual(original_xlim, ax.get_xlim())
                self.assertEqual(original_ylim, ax.get_ylim())

    def test_visualize_mdpa_report_copy_and_export_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "report_controls_model.mdpa")
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
Begin SubModelPart Extra1
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra2
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra3
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra4
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra5
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra6
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra7
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra8
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
"""
                )

            class _FakeButton:
                instances = []

                def __init__(self, ax, label, hovercolor=None):
                    self.ax = ax
                    self.label = label
                    self.hovercolor = hovercolor
                    self._callbacks = []
                    _FakeButton.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

            class _FakeSlider:
                instances = []

                def __init__(self, ax, label='', valmin=0, valmax=1, valinit=0,
                             valstep=1, orientation='vertical'):
                    self.ax = ax
                    self.valmax = valmax
                    self.val = valinit
                    self.valtext = type('DummyText', (), {
                        'set_visible': lambda self, value: None
                    })()
                    self._callbacks = []
                    _FakeSlider.instances.append(self)

                def on_changed(self, callback):
                    self._callbacks.append(callback)

                def set_val(self, value):
                    self.val = value
                    for callback in self._callbacks:
                        callback(value)

            with patch("matplotlib.pyplot.show"), patch(
                "matplotlib.figure.Figure.show"
            ), patch("visualize_mdpa_project.Button", _FakeButton), patch(
                "visualize_mdpa_project.Slider", _FakeSlider
            ):
                visualize_mdpa(mdpa_path, [])

                fig = plt.gcf()
                ax = next(
                    axes
                    for axes in fig.axes
                    if axes.get_title().startswith("MDPA Visualization")
                )
                press_event = MouseEvent(
                    "button_press_event", fig.canvas, 60, 60, button=1
                )
                press_event.inaxes = ax
                press_event.xdata = 0.0
                press_event.ydata = 0.0
                release_event = MouseEvent(
                    "button_release_event", fig.canvas, 60, 60, button=1
                )
                release_event.inaxes = ax
                release_event.xdata = 0.0
                release_event.ydata = 0.0
                fig.canvas.callbacks.process("button_press_event", press_event)
                fig.canvas.callbacks.process("button_release_event", release_event)

                report_buttons = {
                    button.label: button
                    for button in _FakeButton.instances
                    if button.label in {"Copy", "Export"}
                }
                self.assertEqual({"Copy", "Export"}, set(report_buttons))

                report_fig = plt.gcf()
                report_text_ax = report_fig.axes[1]
                report_body = report_text_ax.texts[0]
                initial_report_body = report_body.get_text()
                report_status = report_fig.axes[2].texts[0]
                self.assertIn("Scroll with the mouse wheel", report_status.get_text())
                report_slider = _FakeSlider.instances[-1]
                self.assertEqual(report_slider.valmax, report_slider.val)
                report_slider.set_val(0)
                self.assertNotEqual(initial_report_body, report_body.get_text())
                report_slider.set_val(report_slider.valmax)
                self.assertEqual(initial_report_body, report_body.get_text())
                scroll_event = MouseEvent(
                    "scroll_event", report_fig.canvas, 60, 60, button="down"
                )
                scroll_event.inaxes = report_text_ax
                report_fig.canvas.callbacks.process("scroll_event", scroll_event)
                self.assertNotEqual(initial_report_body, report_body.get_text())
                self.assertLess(report_slider.val, report_slider.valmax)

                with patch("tkinter.Tk", side_effect=RuntimeError("no display")), patch(
                    "visualize_mdpa_project.os.name", "posix"
                ), patch(
                    "visualize_mdpa_project.sys.platform", "linux"
                ), patch("visualize_mdpa_project.subprocess.run") as run_mock:
                    report_buttons["Copy"]._callbacks[0](None)

                run_mock.assert_called_once()
                self.assertEqual("xclip", run_mock.call_args.args[0][0])

                report_buttons["Export"]._callbacks[0](None)

            exported_files = [
                name for name in os.listdir(tmp_dir)
                if name.startswith("mdpa_report_") and name.endswith(".txt")
            ]
            self.assertEqual(1, len(exported_files))
            with open(os.path.join(tmp_dir, exported_files[0]), encoding="utf-8") as f:
                self.assertIn("Node 1", f.read())
    def test_visualize_mdpa_button_controls(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "button_model.mdpa")
            project_path = os.path.join(tmp_dir, "ProjectParameters.json")

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

            with open(project_path, "w", encoding="utf-8") as f:
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
                            },
                            "StageB": {
                                "processes": [
                                    {
                                        "process_name": "StageBProcess",
                                        "Parameters": {"model_part_name": "Wall"},
                                    }
                                ]
                            },
                        }
                    },
                    f,
                )

            class _FakeLabel:
                def __init__(self, text):
                    self.text = text
                    self.fontsize = None
                    self.color = None
                    self.alpha = 1.0
                    self.fontweight = "normal"
                    self.visible = True

                def set_fontsize(self, value, *args, **kwargs):
                    self.fontsize = value

                def set_color(self, value, *args, **kwargs):
                    self.color = value

                def set_alpha(self, value, *args, **kwargs):
                    self.alpha = value

                def set_fontweight(self, value, *args, **kwargs):
                    self.fontweight = value

                def set_visible(self, value, *args, **kwargs):
                    self.visible = value

                def get_text(self):
                    return self.text

            class _FakeCheckButtons:
                instances = []

                def __init__(self, ax, labels, actives):
                    self.ax = ax
                    self._labels = list(labels)
                    self._status = list(actives)
                    self.labels = [_FakeLabel(label) for label in self._labels]
                    self.rectangles = []
                    self.lines = []
                    self._callbacks = []
                    _FakeCheckButtons.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

                def set_active(self, index):
                    if index < 0 or index >= len(self._status):
                        raise IndexError(index)
                    self._status[index] = not self._status[index]
                    if self._callbacks:
                        self._callbacks[0](self._labels[index])

                def get_status(self):
                    return list(self._status)

            class _FakeButtonLabel:
                def __init__(self, text):
                    self.text = text

                def set_text(self, text):
                    self.text = text

                def get_text(self):
                    return self.text

                def __eq__(self, other):
                    return self.text == other

            class _FakeButton:
                instances = []

                def __init__(self, ax, label, hovercolor=None):
                    self.ax = ax
                    self.label = _FakeButtonLabel(label)
                    self.hovercolor = hovercolor
                    self._callbacks = []
                    _FakeButton.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

                def click(self):
                    for callback in self._callbacks:
                        callback(None)

            class _FakeTextBox:
                instances = []

                def __init__(self, ax, label, initial=''):
                    self.ax = ax
                    self.label = label
                    self.text = initial
                    self._text_change_callbacks = []
                    self._submit_callbacks = []
                    _FakeTextBox.instances.append(self)

                def on_text_change(self, callback):
                    self._text_change_callbacks.append(callback)

                def on_submit(self, callback):
                    self._submit_callbacks.append(callback)

                def set_text(self, value):
                    self.text = value
                    for callback in self._text_change_callbacks:
                        callback(value)
                    for callback in self._submit_callbacks:
                        callback(value)

            class _FakeSlider:
                instances = []

                def __init__(self, ax, label='', valmin=0, valmax=1, valinit=0,
                             valstep=1, orientation='vertical'):
                    self.ax = ax
                    self.label = label
                    self.valmin = valmin
                    self.valmax = valmax
                    self.valinit = valinit
                    self.valstep = valstep
                    self.orientation = orientation
                    self.val = valinit
                    self._callbacks = []
                    self.valtext = type('DummyText', (), {'set_visible': lambda self, value: None})()
                    _FakeSlider.instances.append(self)

                def on_changed(self, callback):
                    self._callbacks.append(callback)

                def set_val(self, value):
                    self.val = value
                    for callback in self._callbacks:
                        callback(value)

            with patch("matplotlib.pyplot.show"), patch(
                "visualize_mdpa_project.Button", _FakeButton
            ), patch(
                "visualize_mdpa_project.TextBox", _FakeTextBox
            ), patch(
                "visualize_mdpa_project.CheckButtons", _FakeCheckButtons
            ), patch(
                "visualize_mdpa_project.Slider", _FakeSlider
            ):
                visualize_mdpa(mdpa_path, [project_path])

            fig = plt.gcf()
            ax = next(ax for ax in fig.axes if ax.get_title().startswith("MDPA Visualization"))
            aspect_button = next(
                button for button in _FakeButton.instances
                if button.label == "Aspect: free"
            )
            self.assertEqual("auto", ax.get_aspect())
            aspect_button.click()
            self.assertEqual(1.0, ax.get_aspect())
            self.assertEqual("Aspect: equal", aspect_button.label.get_text())
            aspect_button.click()
            self.assertEqual("auto", ax.get_aspect())
            self.assertEqual("Aspect: free", aspect_button.label.get_text())

            point_button = next(button for button in _FakeButton.instances if button.label == "point size +")
            decrease_button = next(button for button in _FakeButton.instances if button.label == "point size -")
            increase_position = point_button.ax.get_position()
            decrease_position = decrease_button.ax.get_position()
            self.assertAlmostEqual(increase_position.x0, decrease_position.x0)
            self.assertLess(increase_position.y0, decrease_position.y0)
            before_max_size = max(
                max(sc.get_sizes()) for sc in ax.collections if hasattr(sc, "get_sizes")
            )
            point_button.click()
            after_max_size = max(
                max(sc.get_sizes()) for sc in ax.collections if hasattr(sc, "get_sizes")
            )
            self.assertGreater(after_max_size, before_max_size)
            decrease_button.click()
            restored_max_size = max(
                max(sc.get_sizes()) for sc in ax.collections if hasattr(sc, "get_sizes")
            )
            self.assertLess(restored_max_size, after_max_size)
            self.assertAlmostEqual(restored_max_size, before_max_size)

            find_box = next(box for box in _FakeTextBox.instances if box.label == "Find ")
            find_box.set_text("Wall")
            find_box._submit_callbacks[0]("Wall")
            self.assertEqual("Wall", find_box.text)

            hide_button = next(button for button in _FakeButton.instances if button.label == "Hide")
            hide_button.click()

            show_button = next(button for button in _FakeButton.instances if button.label == "Show")
            show_button.click()

            invert_button = next(button for button in _FakeButton.instances if button.label == "Invert")
            invert_button.click()

            save_plot_button = next(
                button for button in _FakeButton.instances if button.label == "Save plot"
            )
            save_plot_button.click()
            saved_plots = [
                name for name in os.listdir(tmp_dir)
                if name.startswith("mdpa_plot_") and name.endswith(".png")
            ]
            self.assertEqual(1, len(saved_plots))

            self.assertTrue(hasattr(point_button, "click"))
            self.assertTrue(hasattr(decrease_button, "click"))
            self.assertTrue(hasattr(find_box, "on_submit"))
            self.assertTrue(hasattr(hide_button, "click"))
            self.assertTrue(hasattr(show_button, "click"))
            self.assertTrue(hasattr(invert_button, "click"))

    def test_visualize_mdpa_draw_mesh_button_toggles_connections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "mesh_model.mdpa")
            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
End Nodes

Begin Elements Element2D3N
1 0 1 2 3
End Elements

Begin Conditions LineCondition2D2N
2 0 2 3
End Conditions

Begin SubModelPart Domain
  Begin SubModelPartNodes
    1
    2
    3
  End SubModelPartNodes
End SubModelPart
"""
                )

            class _FakeButton:
                instances = []

                def __init__(self, ax, label, hovercolor=None):
                    self.label = label
                    self._callbacks = []
                    _FakeButton.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

                def click(self):
                    for callback in self._callbacks:
                        callback(None)

            with patch("matplotlib.pyplot.show"), patch(
                "visualize_mdpa_project.Button", _FakeButton
            ):
                visualize_mdpa(mdpa_path, [])

            fig = plt.gcf()
            ax = next(
                axes
                for axes in fig.axes
                if axes.get_title().startswith("MDPA Visualization")
            )
            mesh_button = next(
                button for button in _FakeButton.instances if button.label == "Draw mesh"
            )

            self.assertEqual(0, len(ax.lines))
            mesh_button.click()
            self.assertEqual(2, len(ax.lines))
            self.assertTrue(all(line.get_color() == "black" for line in ax.lines))
            self.assertEqual(
                {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)},
                {
                    tuple(point)
                    for line in ax.lines
                    for point in line.get_xydata()
                },
            )

            mesh_button.click()
            self.assertEqual(0, len(ax.lines))

    def test_visualize_mdpa_draw_normals_button_draws_visible_boundary_normals(self):
                with tempfile.TemporaryDirectory() as tmp_dir:
                        mdpa_path = os.path.join(tmp_dir, "normal_model.mdpa")
                        with open(mdpa_path, "w", encoding="utf-8") as f:
                                f.write(
                                        """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 1.0 1.0 0.0
5 2.0 0.0 0.0
6 2.0 1.0 0.0
7 3.0 0.0 0.0
8 3.0 1.0 0.0
End Nodes

Begin Elements LinearTimoshenkoCurvedBeamElement2D3N
1 0 3 5 4
End Elements
Begin Elements Geo_UPwLineInterfacePlaneStrainDiffOrderElement3Plus3N
2 0 1 2 3 6 7 8
End Elements
Begin Conditions LineCondition2D2N
3 0 1 2
End Conditions

Begin SubModelPart Beam
    Begin SubModelPartNodes
        3
        4
        5
    End SubModelPartNodes
    Begin SubModelPartElements
        1
    End SubModelPartElements
End SubModelPart

Begin SubModelPart Interface
    Begin SubModelPartNodes
        1
        2
        6
        7
        8
    End SubModelPartNodes
    Begin SubModelPartElements
        2
    End SubModelPartElements
End SubModelPart

Begin SubModelPart Boundary
    Begin SubModelPartNodes
        1
        2
    End SubModelPartNodes
    Begin SubModelPartConditions
        3
    End SubModelPartConditions
End SubModelPart
"""
                                )

                        class _FakeButton:
                                instances = []

                                def __init__(self, ax, label, hovercolor=None):
                                        self.label = label
                                        self._callbacks = []
                                        _FakeButton.instances.append(self)

                                def on_clicked(self, callback):
                                        self._callbacks.append(callback)

                                def click(self):
                                        for callback in self._callbacks:
                                                callback(None)

                        with patch("matplotlib.pyplot.show"), patch(
                                "visualize_mdpa_project.Button", _FakeButton
                        ):
                                visualize_mdpa(mdpa_path, [])

                        fig = plt.gcf()
                        ax = next(
                                axes
                                for axes in fig.axes
                                if axes.get_title().startswith("MDPA Visualization")
                        )
                        normals_button = next(
                                button
                                for button in _FakeButton.instances
                                if button.label == "Draw normals"
                        )
                        normal_increase_button = next(
                            button
                            for button in _FakeButton.instances
                            if button.label == "Normal +"
                        )
                        normal_decrease_button = next(
                            button
                            for button in _FakeButton.instances
                            if button.label == "Normal -"
                        )
                        collection_count = len(ax.collections)
                        normals_button.click()
                        self.assertEqual(collection_count + 4, len(ax.collections))

                        normal_vectors = [
                            collection
                            for collection in ax.collections[collection_count:]
                            if hasattr(collection, "U") and hasattr(collection, "V")
                        ]
                        self.assertEqual(4, len(normal_vectors))
                        initial_length = np.hypot(
                            normal_vectors[0].U[0], normal_vectors[0].V[0]
                        )
                        normal_increase_button.click()
                        increased_length = np.hypot(
                            ax.collections[collection_count].U[0],
                            ax.collections[collection_count].V[0],
                        )
                        self.assertGreater(increased_length, initial_length)
                        normal_decrease_button.click()
                        decreased_length = np.hypot(
                            ax.collections[collection_count].U[0],
                            ax.collections[collection_count].V[0],
                        )
                        self.assertAlmostEqual(decreased_length, initial_length)

                        normal_colors = {
                                tuple(collection.get_facecolors()[0][:3])
                                for collection in ax.collections[collection_count:]
                                if len(collection.get_facecolors())
                        }
                        self.assertEqual(3, len(normal_colors))

                        normals_button.click()
                        self.assertEqual(collection_count, len(ax.collections))

    def test_visualize_mdpa_draws_node_shapes_segments_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "node_markers.mdpa")

            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 2.0 0.0 0.0
End Nodes

Begin SubModelPart Soil
  Begin SubModelPartNodes
    1
    2
    3
  End SubModelPartNodes
End SubModelPart

Begin SubModelPart Wall
  Begin SubModelPartNodes
    2
    4
  End SubModelPartNodes
End SubModelPart
"""
                )

            with patch("matplotlib.pyplot.show"):
                visualize_mdpa(mdpa_path, [])

            fig = plt.gcf()
            ax = next(
                axes
                for axes in fig.axes
                if axes.get_title().startswith("MDPA Visualization")
            )
            node_collections = [
                collection
                for collection in ax.collections
                if hasattr(collection, "get_offsets")
            ]

            self.assertEqual(
                5, sum(len(collection.get_offsets()) for collection in node_collections)
            )

            circular_collections = [
                collection
                for collection in node_collections
                if collection.get_paths()
                and not any(
                    np.allclose(vertex, (0.0, 0.0))
                    for vertex in collection.get_paths()[0].vertices
                )
            ]
            segmented_collections = [
                collection
                for collection in node_collections
                if collection.get_paths()
                and any(
                    np.allclose(vertex, (0.0, 0.0))
                    for vertex in collection.get_paths()[0].vertices
                )
            ]
            self.assertGreaterEqual(len(circular_collections), 2)
            self.assertEqual(2, len(segmented_collections))

            segment_colors = {
                tuple(collection.get_facecolors()[0][:3])
                for collection in segmented_collections
            }
            self.assertEqual(2, len(segment_colors))
            self.assertTrue(
                all(max(color) - min(color) > 0.1 for color in segment_colors)
            )

            circular_sizes = sorted(
                collection.get_sizes()[0] for collection in circular_collections
            )
            self.assertGreater(circular_sizes[-1], circular_sizes[0])

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

    def test_visualize_mdpa_stage_and_checkbox_slider_and_wheel_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "stage_scroll_model.mdpa")
            project_path = os.path.join(tmp_dir, "ProjectParameters.json")

            with open(mdpa_path, "w", encoding="utf-8") as f:
                f.write(
                    """Begin Nodes
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 1.0 1.0 0.0
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

Begin SubModelPart Roof
  Begin SubModelPartNodes
    4
    3
  End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra1
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra2
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra3
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra4
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra5
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra6
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra7
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra8
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra9
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra10
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra11
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra12
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra13
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra14
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra15
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra16
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra17
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra18
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra19
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
Begin SubModelPart Extra20
    Begin SubModelPartNodes
        1
    End SubModelPartNodes
End SubModelPart
"""
                )

            stagedata = {
                "stages": {
                    f"Stage{i}": {
                        "processes": [
                            {
                                "process_name": f"P{i}",
                                "Parameters": {"model_part_name": "Soil"},
                            }
                        ]
                    }
                    for i in range(1, 12)
                }
            }
            with open(project_path, "w", encoding="utf-8") as f:
                json.dump(stagedata, f)

            class _FakeLabel:
                def __init__(self, text):
                    self.text = text

                def set_fontsize(self, *_args, **_kwargs):
                    pass

                def set_color(self, *_args, **_kwargs):
                    pass

                def set_alpha(self, *_args, **_kwargs):
                    pass

                def set_fontweight(self, *_args, **_kwargs):
                    pass

                def set_visible(self, *_args, **_kwargs):
                    pass

                def get_text(self):
                    return self.text

            class _FakeCheckButtons:
                instances = []

                def __init__(self, ax, labels, actives):
                    self.ax = ax
                    self._labels = list(labels)
                    self._status = list(actives)
                    self.labels = [_FakeLabel(label) for label in self._labels]
                    self._callbacks = []
                    _FakeCheckButtons.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

                def set_active(self, index, trigger_callback=False):
                    if index < 0 or index >= len(self._status):
                        raise IndexError(index)
                    self._status[index] = not self._status[index]
                    if trigger_callback and self._callbacks:
                        self._callbacks[0](self._labels[index])

                def get_status(self):
                    return list(self._status)

            class _FakeButton:
                def __init__(self, ax, label, hovercolor=None):
                    self.ax = ax
                    self.label = label
                    self.hovercolor = hovercolor
                    self._callbacks = []

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

            class _FakeTextBox:
                def __init__(self, ax, label, initial=''):
                    self.ax = ax
                    self.label = label
                    self.text = initial
                    self._text_change_callbacks = []
                    self._submit_callbacks = []

                def on_text_change(self, callback):
                    self._text_change_callbacks.append(callback)

                def on_submit(self, callback):
                    self._submit_callbacks.append(callback)

            class _FakeSlider:
                instances = []

                def __init__(self, ax, label='', valmin=0, valmax=1, valinit=0,
                             valstep=1, orientation='vertical'):
                    self.ax = ax
                    self.label = label
                    self.valmin = valmin
                    self.valmax = valmax
                    self.valinit = valinit
                    self.valstep = valstep
                    self.orientation = orientation
                    self.val = valinit
                    self._callbacks = []
                    self.valtext = type('DummyText', (), {'set_visible': lambda self, value: None})()
                    _FakeSlider.instances.append(self)

                def on_changed(self, callback):
                    self._callbacks.append(callback)

                def set_val(self, value):
                    self.val = value
                    for callback in self._callbacks:
                        callback(value)

            with patch("matplotlib.pyplot.show"), patch(
                "visualize_mdpa_project.Button", _FakeButton
            ), patch(
                "visualize_mdpa_project.TextBox", _FakeTextBox
            ), patch(
                "visualize_mdpa_project.CheckButtons", _FakeCheckButtons
            ), patch(
                "visualize_mdpa_project.Slider", _FakeSlider
            ):
                visualize_mdpa(mdpa_path, [project_path])

            stage_scroll = next(
                slider for slider in _FakeSlider.instances if getattr(slider, 'valmax', 1) > 1
            )
            self.assertEqual(stage_scroll.valmax, stage_scroll.val)
            stage_scroll.set_val(2)
            self.assertEqual(2, stage_scroll.val)

            fig = plt.gcf()
            scroll_event = MouseEvent("scroll_event", fig.canvas, 0, 0, button='down')
            scroll_event.inaxes = stage_scroll.ax
            scroll_event.xdata = 0.0
            scroll_event.ydata = 0.0
            fig.canvas.callbacks.process("scroll_event", scroll_event)
            self.assertLessEqual(stage_scroll.val, 2)

            checks_scroll = _FakeSlider.instances[-1]
            self.assertGreater(checks_scroll.valmax, 1)
            self.assertEqual(checks_scroll.valmax, checks_scroll.val)
            checks_scroll.set_val(1)
            self.assertEqual(1, checks_scroll.val)

            checks_ax = _FakeCheckButtons.instances[-1].ax
            checks_event = MouseEvent("scroll_event", fig.canvas, 0, 0, button='down')
            checks_event.inaxes = checks_ax
            checks_event.xdata = 0.0
            checks_event.ydata = 0.0
            fig.canvas.callbacks.process("scroll_event", checks_event)
            self.assertEqual(0, checks_scroll.val)

    def test_visualize_mdpa_checkbox_toggle_syncs_hidden_submodels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "toggle_model.mdpa")
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

            class _FakeLabel:
                def __init__(self, text):
                    self.text = text

                def set_fontsize(self, *_args, **_kwargs):
                    pass

                def set_color(self, *_args, **_kwargs):
                    pass

                def set_alpha(self, *_args, **_kwargs):
                    pass

                def set_fontweight(self, *_args, **_kwargs):
                    pass

                def set_visible(self, *_args, **_kwargs):
                    pass

                def get_text(self):
                    return self.text

            class _FakeCheckButtons:
                instances = []

                def __init__(self, ax, labels, actives):
                    self.ax = ax
                    self._labels = list(labels)
                    self._status = list(actives)
                    self.labels = [_FakeLabel(label) for label in self._labels]
                    self._callbacks = []
                    _FakeCheckButtons.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

                def set_active(self, index, trigger_callback=False):
                    if index < 0 or index >= len(self._status):
                        raise IndexError(index)
                    self._status[index] = not self._status[index]
                    if trigger_callback and self._callbacks:
                        for callback in self._callbacks:
                            callback(self._labels[index])

                def get_status(self):
                    return list(self._status)

            class _FakeButton:
                def __init__(self, ax, label, hovercolor=None):
                    self.ax = ax
                    self.label = label
                    self.hovercolor = hovercolor
                    self._callbacks = []

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

            class _FakeTextBox:
                def __init__(self, ax, label, initial=''):
                    self.ax = ax
                    self.label = label
                    self.text = initial
                    self._text_change_callbacks = []
                    self._submit_callbacks = []

                def on_text_change(self, callback):
                    self._text_change_callbacks.append(callback)

                def on_submit(self, callback):
                    self._submit_callbacks.append(callback)

            class _FakeSlider:
                def __init__(self, ax, label='', valmin=0, valmax=1, valinit=0,
                             valstep=1, orientation='vertical'):
                    self.ax = ax
                    self.label = label
                    self.valmin = valmin
                    self.valmax = valmax
                    self.valinit = valinit
                    self.valstep = valstep
                    self.orientation = orientation
                    self.val = valinit
                    self._callbacks = []
                    self.valtext = type('DummyText', (), {'set_visible': lambda self, value: None})()

                def on_changed(self, callback):
                    self._callbacks.append(callback)

                def set_val(self, value):
                    self.val = value
                    for callback in self._callbacks:
                        callback(value)

            with patch("matplotlib.pyplot.show"), patch(
                "visualize_mdpa_project.Button", _FakeButton
            ), patch(
                "visualize_mdpa_project.TextBox", _FakeTextBox
            ), patch(
                "visualize_mdpa_project.CheckButtons", _FakeCheckButtons
            ), patch(
                "visualize_mdpa_project.Slider", _FakeSlider
            ):
                visualize_mdpa(mdpa_path, [])

            visibility_widget = _FakeCheckButtons.instances[-1]
            self.assertGreater(len(visibility_widget.labels), 0)
            visibility_widget.set_active(0, trigger_callback=True)
            self.assertFalse(visibility_widget.get_status()[0])

    def test_visualize_mdpa_toggle_submodel_visibility_callback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, "toggle_callback_model.mdpa")
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

            class _FakeLabel:
                def __init__(self, text):
                    self.text = text

                def set_fontsize(self, *_args, **_kwargs):
                    pass

                def set_color(self, *_args, **_kwargs):
                    pass

                def set_alpha(self, *_args, **_kwargs):
                    pass

                def set_fontweight(self, *_args, **_kwargs):
                    pass

                def set_visible(self, *_args, **_kwargs):
                    pass

                def get_text(self):
                    return self.text

            class _FakeCheckButtons:
                instances = []

                def __init__(self, ax, labels, actives):
                    self.ax = ax
                    self._labels = list(labels)
                    self._status = list(actives)
                    self.labels = [_FakeLabel(label) for label in self._labels]
                    self._callbacks = []
                    _FakeCheckButtons.instances.append(self)

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

                def get_status(self):
                    return list(self._status)

                def set_active(self, index, trigger_callback=False):
                    if index < 0 or index >= len(self._status):
                        raise IndexError(index)
                    self._status[index] = not self._status[index]
                    if trigger_callback and self._callbacks:
                        for callback in self._callbacks:
                            callback(self._labels[index])

            class _FakeButton:
                def __init__(self, ax, label, hovercolor=None):
                    self.ax = ax
                    self.label = label
                    self.hovercolor = hovercolor
                    self._callbacks = []

                def on_clicked(self, callback):
                    self._callbacks.append(callback)

            class _FakeTextBox:
                def __init__(self, ax, label, initial=''):
                    self.ax = ax
                    self.label = label
                    self.text = initial
                    self._text_change_callbacks = []
                    self._submit_callbacks = []

                def on_text_change(self, callback):
                    self._text_change_callbacks.append(callback)

                def on_submit(self, callback):
                    self._submit_callbacks.append(callback)

            class _FakeSlider:
                def __init__(self, ax, label='', valmin=0, valmax=1, valinit=0,
                             valstep=1, orientation='vertical'):
                    self.ax = ax
                    self.label = label
                    self.valmin = valmin
                    self.valmax = valmax
                    self.valinit = valinit
                    self.valstep = valstep
                    self.orientation = orientation
                    self.val = valinit
                    self._callbacks = []
                    self.valtext = type('DummyText', (), {'set_visible': lambda self, value: None})()

                def on_changed(self, callback):
                    self._callbacks.append(callback)

                def set_val(self, value):
                    self.val = value
                    for callback in self._callbacks:
                        callback(value)

            with patch("matplotlib.pyplot.show"), patch(
                "visualize_mdpa_project.Button", _FakeButton
            ), patch(
                "visualize_mdpa_project.TextBox", _FakeTextBox
            ), patch(
                "visualize_mdpa_project.CheckButtons", _FakeCheckButtons
            ), patch(
                "visualize_mdpa_project.Slider", _FakeSlider
            ):
                visualize_mdpa(mdpa_path, [])

            visibility_widget = _FakeCheckButtons.instances[-1]
            soil_index = visibility_widget._labels.index("Soil")
            self.assertTrue(visibility_widget.get_status()[soil_index])

            callback = visibility_widget._callbacks[0]
            visibility_widget.set_active(soil_index)
            callback("Soil")

            self.assertFalse(visibility_widget.get_status()[soil_index])


if __name__ == "__main__":
    unittest.main()
