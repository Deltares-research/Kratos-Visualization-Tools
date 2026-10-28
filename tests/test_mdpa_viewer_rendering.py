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
import numpy as np

from visualize_mdpa_project import (
    visualize_mdpa,
)


class TestVisualizeMdpaLite(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

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


if __name__ == "__main__":
    unittest.main()
