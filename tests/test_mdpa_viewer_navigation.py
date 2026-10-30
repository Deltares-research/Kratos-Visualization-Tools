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
    _resolve_constraint_relations,
    _build_face_separation,
    _separated_coordinates,
    _normalize_submodelpart_name,
    _normal_segments,
    _stage_from_process_label,
    _connection_paths,
    _normal_vector,
    _resolve_stage_activity,
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
