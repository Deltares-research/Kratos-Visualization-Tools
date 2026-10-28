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

from visualize_mdpa_project import (
    visualize_mdpa,
)


class TestVisualizeMdpaLite(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

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
