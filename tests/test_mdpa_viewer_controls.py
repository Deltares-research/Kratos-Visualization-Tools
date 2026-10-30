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

    def test_visualize_button_filters_stage_activity_and_operations(self):
        from matplotlib.collections import PathCollection
        from matplotlib.widgets import Button as RealButton
        from matplotlib.widgets import CheckButtons as RealCheckButtons

        buttons = []
        checks = []

        def capture_button(*args, **kwargs):
            widget = RealButton(*args, **kwargs)
            buttons.append(widget)
            return widget

        def capture_checks(*args, **kwargs):
            widget = RealCheckButtons(*args, **kwargs)
            checks.append(widget)
            return widget

        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, 'activity.mdpa')
            project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
            with open(mdpa_path, 'w', encoding='utf-8') as output:
                output.write('''Begin Nodes
1 0 0 0
2 1 0 0
3 2 0 0
4 3 0 0
End Nodes
Begin SubModelPart ActivePart
Begin SubModelPartNodes
1
End SubModelPartNodes
End SubModelPart
Begin SubModelPart SwitchPart
Begin SubModelPartNodes
2
End SubModelPartNodes
End SubModelPart
Begin SubModelPart OffPart
Begin SubModelPartNodes
3
End SubModelPartNodes
End SubModelPart
Begin SubModelPart ProcessPart
Begin SubModelPartNodes
4
End SubModelPartNodes
End SubModelPart
''')
            def operation(name, part):
                return {
                    'name': f'Operations.KratosMultiphysics.{name}ModelPartOperation',
                    'Parameters': {'model_part_name': part},
                }

            project = {'stages': {
                'Stage1': {
                    'solver_settings': {
                        'problem_domain_sub_model_part_list':
                            ['Domain.ActivePart', 'Domain.SwitchPart', 'Domain.OffPart']},
                    'stage_preprocess': {'operations': [
                        operation('Deactivate', 'Domain.SwitchPart'),
                        operation('Deactivate', 'Domain.OffPart'),
                    ]},
                },
                'Stage2': {
                    'stage_preprocess': {'operations': [
                        operation('Activate', 'Domain.SwitchPart'),
                    ]},
                    'processes': [{
                        'process_name': 'ApplyVectorConstraintTableProcess',
                        'Parameters': {'model_part_name_list': [
                            'Domain.ProcessPart', 'Domain.OffPart']},
                    }],
                },
                'Stage3': {'stage_preprocess': {
                    'operations': [
                        operation('Deactivate', 'Domain.SwitchPart'),
                        operation('Activate', 'Domain.SwitchPart'),
                        operation('Deactivate', 'Domain.ActivePart'),
                    ],
                    'modelers': [{
                        'name': 'Modeler',
                        'Parameters': {'model_part_name': 'Domain.ProcessPart'},
                    }],
                }},
            }}
            with open(project_path, 'w', encoding='utf-8') as output:
                json.dump(project, output)

            with patch('matplotlib.pyplot.show'), patch('builtins.print'), patch(
                'visualize_mdpa_project.Button', side_effect=capture_button
            ), patch(
                'visualize_mdpa_project.CheckButtons', side_effect=capture_checks
            ):
                visualize_mdpa(mdpa_path, [project_path])
                figure = plt.gcf()
                axes = next(axis for axis in figure.axes
                            if axis.get_title().startswith('MDPA Visualization'))
                controls = figure._mdpa_diagnostic_widgets

                def points():
                    return {
                        tuple(point)
                        for collection in axes.collections
                        if isinstance(collection, PathCollection)
                        for point in collection.get_offsets()
                    }

                def select_stage(stage):
                    widget = next(widget for widget in reversed(checks)
                                  if 'All' in [label.get_text() for label in widget.labels]
                                  and stage in [label.get_text() for label in widget.labels])
                    widget.set_active([label.get_text() for label in widget.labels].index(stage))

                all_points = {(0, 0), (1, 0), (2, 0)}
                all_stage_points = all_points | {(3, 0)}
                self.assertEqual('Visualize: all', controls['visualize'].label.get_text())
                self.assertEqual(all_stage_points, points())
                controls['visualize']._observers.process('clicked', None)
                self.assertEqual(all_stage_points, points())

                select_stage('Stage1')
                self.assertEqual({(0, 0)}, points())
                stage_checks = next(widget for widget in reversed(checks)
                                    if 'ActivePart' in [label.get_text()
                                                       for label in widget.labels])
                active_index = [label.get_text() for label in stage_checks.labels].index(
                    'ActivePart')
                stage_checks.set_active(active_index)
                self.assertEqual(set(), points())
                stage_checks = next(widget for widget in reversed(checks)
                                    if 'ActivePart' in [label.get_text()
                                                       for label in widget.labels])
                active_index = [label.get_text() for label in stage_checks.labels].index(
                    'ActivePart')
                stage_checks.set_active(active_index)
                self.assertEqual({(0, 0)}, points())
                controls['visualize']._observers.process('clicked', None)
                self.assertEqual({(1, 0), (2, 0)}, points())
                controls['visualize']._observers.process('clicked', None)
                self.assertEqual(all_points, points())

                select_stage('Stage2')
                self.assertEqual(all_stage_points, points())
                controls['visualize']._observers.process('clicked', None)
                self.assertEqual({(0, 0), (1, 0), (3, 0)}, points())
                controls['visualize']._observers.process('clicked', None)
                self.assertEqual({(2, 0)}, points())
                controls['visualize']._observers.process('clicked', None)

                select_stage('Stage3')
                self.assertEqual(all_stage_points, points())
                controls['visualize']._observers.process('clicked', None)
                self.assertEqual({(1, 0), (3, 0)}, points())
                controls['visualize']._observers.process('clicked', None)
                self.assertEqual({(0, 0), (2, 0)}, points())

                controls['visualize']._observers.process('clicked', None)
                for process_visible in (True, False):
                    if not process_visible:
                        widget = next(widget for widget in reversed(checks)
                                      if 'ProcessPart' in [
                                          label.get_text() for label in widget.labels])
                        widget.set_active([label.get_text() for label in widget.labels].index(
                            'ProcessPart'))
                    for stage in ('Stage1', 'Stage2', 'Stage3'):
                        select_stage(stage)
                        all_visible = points()
                        if stage != 'Stage1':
                            self.assertEqual(process_visible, (3, 0) in all_visible)
                        controls['visualize']._observers.process('clicked', None)
                        active_visible = points()
                        controls['visualize']._observers.process('clicked', None)
                        deactivated_visible = points()
                        self.assertEqual(all_visible - deactivated_visible, active_visible)
                        self.assertFalse(active_visible & deactivated_visible)
                        controls['visualize']._observers.process('clicked', None)

                show_button = next(widget for widget in buttons
                                   if widget.label.get_text() == 'Show')
                show_button._observers.process('clicked', None)
                self.assertEqual(all_stage_points, points())
                show_button._observers.process('clicked', None)
                self.assertEqual(all_stage_points, points())

                controls['visualize']._observers.process('clicked', None)
                controls['visualize']._observers.process('clicked', None)
                select_stage('All')
                self.assertEqual(all_stage_points, points())
                self.assertEqual('Visualize: deactivated',
                                 controls['visualize'].label.get_text())

    def test_viewer_shortcuts_do_not_intercept_text_box_key_events(self):
        from matplotlib.widgets import TextBox as RealTextBox

        text_boxes = []

        def capture_text_box(*args, **kwargs):
            widget = RealTextBox(*args, **kwargs)
            text_boxes.append(widget)
            return widget

        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, 'keyboard.mdpa')
            project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
            with open(mdpa_path, 'w', encoding='utf-8') as output:
                output.write('''Begin Nodes
1 0 0 0
2 1 1 0
End Nodes
Begin SubModelPart Soil
Begin SubModelPartNodes
1
2
End SubModelPartNodes
End SubModelPart
''')
            with open(project_path, 'w', encoding='utf-8') as output:
                json.dump({}, output)
            with patch('matplotlib.pyplot.show'), patch('builtins.print') as print_mock, patch(
                'visualize_mdpa_project.TextBox', side_effect=capture_text_box
            ):
                visualize_mdpa(mdpa_path, [project_path])
                figure = plt.gcf()
                canvas = figure.canvas
                canvas.draw()
                self.assertEqual(3, len(text_boxes))
                for widget in text_boxes:
                    with self.subTest(field=widget.label.get_text()):
                        x, y = widget.ax.transAxes.transform((0.5, 0.5))
                        for name in ('button_press_event', 'button_release_event'):
                            canvas.callbacks.process(
                                name, MouseEvent(name, canvas, x, y, button=1))
                        self.assertTrue(widget.capturekeystrokes)
                        print_mock.reset_mock()
                        for key in ('1', 'z', 'r'):
                            before = widget.text
                            canvas.callbacks.process(
                                'key_press_event', KeyEvent('key_press_event', canvas, key=key))
                            self.assertEqual(len(before) + 1, len(widget.text))
                            self.assertIn(key, widget.text)
                        print_mock.assert_not_called()

                        x, y = figure.transFigure.transform((0.99, 0.99))
                        for name in ('button_press_event', 'button_release_event'):
                            canvas.callbacks.process(
                                name, MouseEvent(name, canvas, x, y, button=1))
                        self.assertFalse(any(box.capturekeystrokes for box in text_boxes))
                        print_mock.reset_mock()
                        canvas.callbacks.process(
                            'key_press_event', KeyEvent('key_press_event', canvas, key='1'))
                        self.assertTrue(any('SubModelPart #1: Soil' in str(call.args)
                                            for call in print_mock.call_args_list))

    def test_refresh_lifecycle_rebuilds_only_for_visibility_and_activation(self):
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgba
        from matplotlib.quiver import Quiver
        from matplotlib.widgets import Button as RealButton
        from matplotlib.widgets import CheckButtons as RealCheckButtons

        buttons = []
        checks = []

        def capture_buttons(*args, **kwargs):
            widget = RealButton(*args, **kwargs)
            buttons.append(widget)
            return widget

        def capture_checks(*args, **kwargs):
            widget = RealCheckButtons(*args, **kwargs)
            checks.append(widget)
            return widget

        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, 'lifecycle.mdpa')
            project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
            with open(mdpa_path, 'w', encoding='utf-8') as output:
                output.write('''Begin Nodes
1 0 0 0
2 2 0 0
3 0 0 0
4 2 0 0
5 1 1 0
End Nodes
Begin Elements Interface2Plus2N
10 0 1 2 3 4
End Elements
Begin Elements UPwSmallStrainElement2D3N
11 0 3 4 5
End Elements
Begin SubModelPart Soil
Begin SubModelPartNodes
1
2
3
4
5
End SubModelPartNodes
Begin SubModelPartElements
10
End SubModelPartElements
End SubModelPart
''')
            with open(project_path, 'w', encoding='utf-8') as output:
                json.dump({'stages': {
                    name: {'solver_settings': {'problem_domain_sub_model_part_list': ['Soil']}}
                    for name in ('Before', 'After')
                }}, output)
            with patch('matplotlib.pyplot.show'), patch('builtins.print'), patch(
                'visualize_mdpa_project.Button', side_effect=capture_buttons
            ), patch(
                'visualize_mdpa_project.CheckButtons', side_effect=capture_checks
            ), patch(
                'visualize_mdpa_project._build_face_separation', wraps=_build_face_separation
            ) as rebuild, patch(
                'visualize_mdpa_project._separated_coordinates', wraps=_separated_coordinates
            ) as coordinates:
                visualize_mdpa(mdpa_path, [project_path])
                figure = plt.gcf()
                axes = next(axis for axis in figure.axes
                            if axis.get_title().startswith('MDPA Visualization'))
                controls = figure._mdpa_diagnostic_widgets
                separation_button = controls['separated_faces']
                inactive_color = to_rgba('#e6e6e6')
                active_color = to_rgba('#a8d5a2')
                np.testing.assert_allclose(separation_button.ax.get_facecolor(), inactive_color)
                self.assertEqual({('Elements', 10)},
                                 rebuild.call_args.kwargs['visible_interfaces'])

                def click(label):
                    widget = next(widget for widget in buttons
                                  if widget.label.get_text() == label)
                    widget._observers.process('clicked', None)

                def toggle(label):
                    widget = next(widget for widget in reversed(checks)
                                  if label in [text.get_text() for text in widget.labels])
                    widget.set_active([text.get_text() for text in widget.labels].index(label))

                def refresh(action, *, builds, updates=1):
                    rebuild.reset_mock()
                    coordinates.reset_mock()
                    action()
                    self.assertEqual(builds, rebuild.call_count)
                    self.assertEqual(updates, coordinates.call_count)

                def snapshot():
                    return (
                        sorted(tuple(point) for collection in axes.collections
                               if isinstance(collection, PathCollection)
                               for point in collection.get_offsets()),
                        sorted((line.get_color(), line.get_linestyle(),
                                tuple(line.get_xdata()), tuple(line.get_ydata()))
                               for line in axes.lines),
                        sorted((text.get_text(), tuple(text.xy)) for text in axes.texts),
                        sorted((tuple(collection.get_offsets().ravel()),
                                tuple(collection.U), tuple(collection.V))
                               for collection in axes.collections
                               if isinstance(collection, Quiver)),
                    )

                refresh(lambda: click('Draw mesh'), builds=0, updates=0)
                refresh(lambda: click('Draw normals'), builds=0, updates=0)
                controls['node_id'].set_val('3')
                original = snapshot()
                self.assertTrue(original[3])
                refresh(lambda: click('Separated Faces'), builds=1)
                np.testing.assert_allclose(separation_button.ax.get_facecolor(), active_color)
                separated = snapshot()
                self.assertNotEqual(original, separated)
                refresh(lambda: click('Gap +'), builds=0)
                self.assertNotEqual(separated, snapshot())
                refresh(lambda: click('Gap -'), builds=0)
                restored_gap = snapshot()
                refresh(lambda: click('point size +'), builds=0, updates=0)
                self.assertEqual(restored_gap, snapshot())
                refresh(lambda: toggle('Soil'), builds=1)
                self.assertEqual(([], [], [], []), snapshot())
                refresh(lambda: toggle('Soil'), builds=1)
                self.assertEqual(restored_gap, snapshot())
                for label in ('Hide', 'Show', 'Invert', 'Invert'):
                    refresh(lambda: click(label), builds=1)
                self.assertEqual(restored_gap, snapshot())
                for stage in ('Before', 'After', 'All'):
                    refresh(lambda: toggle(stage), builds=1)
                    self.assertEqual(restored_gap, snapshot())
                refresh(lambda: click('Separated Faces'), builds=0)
                np.testing.assert_allclose(separation_button.ax.get_facecolor(), inactive_color)
                self.assertEqual(original, snapshot())
                refresh(lambda: controls['gap_value'].set_val('0.1'), builds=0, updates=0)
                self.assertEqual(original, snapshot())
                refresh(lambda: click('Separated Faces'), builds=1)
                np.testing.assert_allclose(separation_button.ax.get_facecolor(), active_color)
                self.assertNotEqual(original, snapshot())

    def test_first_separation_activation_uses_visible_interfaces(self):
        from matplotlib.collections import PathCollection
        from matplotlib.widgets import CheckButtons as RealCheckButtons

        checks = []

        def capture_checks(*args, **kwargs):
            widget = RealCheckButtons(*args, **kwargs)
            checks.append(widget)
            return widget

        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, 'partly_visible.mdpa')
            project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
            with open(mdpa_path, 'w', encoding='utf-8') as output:
                output.write('''Begin Nodes
1 0 0 0
2 2 0 0
3 0 0 0
4 2 0 0
5 1 1 0
End Nodes
Begin Elements Interface2Plus2N
10 0 1 2 3 4
End Elements
Begin Elements UPwSmallStrainElement2D3N
11 0 3 4 5
End Elements
Begin SubModelPart VisibleEnd
Begin SubModelPartNodes
1
3
End SubModelPartNodes
End SubModelPart
''')
            with open(project_path, 'w', encoding='utf-8') as output:
                json.dump({}, output)
            parsed = parse_mdpa_data(mdpa_path)
            _, _, unfiltered_directions, _ = _build_face_separation(parsed)
            self.assertGreater(np.linalg.norm(unfiltered_directions[3]), 0)

            with patch('matplotlib.pyplot.show'), patch('builtins.print'), patch(
                'visualize_mdpa_project.CheckButtons', side_effect=capture_checks
            ):
                visualize_mdpa(mdpa_path, [project_path])
                figure = plt.gcf()
                axes = next(axis for axis in figure.axes
                            if axis.get_title().startswith('MDPA Visualization'))
                controls = figure._mdpa_diagnostic_widgets

                def assert_unshifted():
                    offsets = [point for collection in axes.collections
                               if isinstance(collection, PathCollection)
                               for point in collection.get_offsets()]
                    self.assertTrue(offsets)
                    np.testing.assert_allclose(offsets, np.zeros((len(offsets), 2)))
                    self.assertFalse(any(line.get_linestyle() == '--' for line in axes.lines))

                controls['separated_faces']._observers.process('clicked', None)
                assert_unshifted()
                for _ in range(2):
                    widget = checks[-1]
                    index = [label.get_text() for label in widget.labels].index('VisibleEnd')
                    widget.set_active(index)
                assert_unshifted()
                controls['separated_faces']._observers.process('clicked', None)
                controls['separated_faces']._observers.process('clicked', None)
                assert_unshifted()

    def test_ungrouped_interface_links_follow_entity_visibility(self):
        from matplotlib.widgets import CheckButtons as RealCheckButtons

        for with_constraint in (False, True):
            with self.subTest(with_constraint=with_constraint), tempfile.TemporaryDirectory() as tmp_dir:
                mdpa_path = os.path.join(tmp_dir, 'ungrouped.mdpa')
                project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
                with open(mdpa_path, 'w', encoding='utf-8') as output:
                    output.write('''Begin Nodes
1 0 0 0
2 2 0 0
3 0 0 0
4 2 0 0
End Nodes
Begin Elements Interface2Plus2N
10 0 1 2 3 4
End Elements
''')
                    for name, node_ids in (
                        ('Master', (1, 2)), ('Slave', (3, 4)), ('OtherEnd', (2, 4))
                    ):
                        output.write(f'Begin SubModelPart {name}\nBegin SubModelPartNodes\n')
                        output.writelines(f'{node_id}\n' for node_id in node_ids)
                        output.write('End SubModelPartNodes\nEnd SubModelPart\n')
                with open(project_path, 'w', encoding='utf-8') as output:
                    json.dump({'processes': [{'process_name': 'Constraints', 'Parameters': {
                        'master_model_part_name': 'Master',
                        'slave_model_part_name': 'Slave'}}]} if with_constraint else {}, output)
                checks = []

                def capture_checks(*args, **kwargs):
                    widget = RealCheckButtons(*args, **kwargs)
                    checks.append(widget)
                    return widget

                with patch('matplotlib.pyplot.show'), patch('builtins.print') as print_mock, patch(
                    'visualize_mdpa_project.CheckButtons', side_effect=capture_checks
                ):
                    visualize_mdpa(mdpa_path, [project_path])
                    figure = plt.gcf()
                    axes = next(axis for axis in figure.axes
                                if axis.get_title().startswith('MDPA Visualization'))
                    controls = figure._mdpa_diagnostic_widgets
                    controls['separated_faces']._observers.process('clicked', None)

                    def links():
                        return [line for line in axes.lines if line.get_linestyle() == '--']

                    def toggle(name):
                        widget = next(widget for widget in reversed(checks)
                                      if name in [label.get_text() for label in widget.labels])
                        index = [label.get_text() for label in widget.labels].index(name)
                        widget.set_active(index)

                    self.assertEqual(2, len(links()))
                    self.assertTrue(all(line.get_color() == '#b64b24' for line in links()))
                    self.assertTrue(all(not np.allclose(
                        (line.get_xdata()[0], line.get_ydata()[0]),
                        (line.get_xdata()[1], line.get_ydata()[1])) for line in links()))
                    controls['node_id'].set_val('3')
                    report = str(print_mock.call_args.args[0])
                    self.assertIn("Interface entities shown/referenced: [('Elements', 10)]", report)
                    self.assertIn('Interface parts shown/referenced: None', report)
                    toggle('Slave')
                    self.assertEqual([], links())
                    toggle('Slave')
                    self.assertEqual(2, len(links()))
                    self.assertTrue(all(line.get_color() == '#b64b24' for line in links()))

                    # Hide one entity membership while retaining visible nodes.
                    # With constraints, the link then becomes constraint-only.
                    parsed = parse_mdpa_data(mdpa_path)
                    parsed.entity_submodels[('Elements', 10)].add('OtherEnd')
                    plt.close('all')
                    with patch('visualize_mdpa_project.parse_mdpa_data',
                               return_value=parsed):
                        visualize_mdpa(mdpa_path, [project_path])
                    figure = plt.gcf()
                    axes = next(axis for axis in figure.axes
                                if axis.get_title().startswith('MDPA Visualization'))
                    figure._mdpa_diagnostic_widgets['separated_faces']._observers.process('clicked', None)
                    toggle('OtherEnd')
                    self.assertEqual(2 if with_constraint else 0, len(links()))
                    self.assertTrue(all(line.get_color() == '#167d8d' for line in links()))

    def test_unsupported_separation_geometry_does_not_block_viewer_or_visibility(self):
        from matplotlib.widgets import Button as RealButton
        from matplotlib.widgets import CheckButtons as RealCheckButtons

        buttons = []
        checks = []

        def capture_buttons(*args, **kwargs):
            widget = RealButton(*args, **kwargs)
            buttons.append(widget)
            return widget

        def capture_checks(*args, **kwargs):
            widget = RealCheckButtons(*args, **kwargs)
            checks.append(widget)
            return widget

        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, 'unsupported.mdpa')
            project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
            with open(mdpa_path, 'w', encoding='utf-8') as output:
                output.write('''Begin Nodes
1 0 0 0
2 0 0 0
3 1 1 0
4 2 0 0
End Nodes
Begin Elements CrBeamElement2D3N
10 0 2 3 4
End Elements
Begin Conditions LineCondition2D2N
20 0 2 4
End Conditions
''')
                for name, node_ids in (('Master', (1,)), ('Slave', (2, 3, 4))):
                    output.write(f'Begin SubModelPart {name}\nBegin SubModelPartNodes\n')
                    output.writelines(f'{node_id}\n' for node_id in node_ids)
                    output.write('End SubModelPartNodes\nEnd SubModelPart\n')
            with open(project_path, 'w', encoding='utf-8') as output:
                json.dump({'processes': [{'process_name': 'Constraints', 'Parameters': {
                    'master_model_part_name': 'Master',
                    'slave_model_part_name': 'Slave'}}]}, output)
            with patch('matplotlib.pyplot.show'), patch('builtins.print') as print_mock, patch(
                'visualize_mdpa_project.Button', side_effect=capture_buttons
            ), patch('visualize_mdpa_project.CheckButtons', side_effect=capture_checks):
                visualize_mdpa(mdpa_path, [project_path])
                figure = plt.gcf()
                controls = figure._mdpa_diagnostic_widgets
                controls['node_id'].set_val('2')
                self.assertTrue(any('Unsupported separation geometry' in str(call.args)
                                    and 'CrBeamElement2D3N' in str(call.args)
                                    for call in print_mock.call_args_list))
                controls['separated_faces']._observers.process('clicked', None)
                for label in ('Hide', 'Show', 'Invert', 'Invert'):
                    button = next(widget for widget in buttons
                                  if widget.label.get_text() == label)
                    button._observers.process('clicked', None)
                for _ in range(2):
                    widget = next(widget for widget in reversed(checks)
                                  if 'Slave' in [label.get_text() for label in widget.labels])
                    index = [label.get_text() for label in widget.labels].index('Slave')
                    widget.set_active(index)
                controls['separated_faces']._observers.process('clicked', None)
                figure.canvas.draw()

    def test_separation_reacts_to_interface_visibility_and_stages(self):
        from matplotlib.collections import PathCollection
        from matplotlib.widgets import Button as RealButton
        from matplotlib.widgets import CheckButtons as RealCheckButtons

        checks = []
        buttons = []

        def capture_checks(*args, **kwargs):
            widget = RealCheckButtons(*args, **kwargs)
            checks.append(widget)
            return widget

        def capture_buttons(*args, **kwargs):
            widget = RealButton(*args, **kwargs)
            buttons.append(widget)
            return widget

        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, 'corner.mdpa')
            project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
            with open(mdpa_path, 'w', encoding='utf-8') as output:
                output.write('''Begin Nodes
1 0 0 0
2 -0.1 1 0
3 0.1 1 0
4 0 0 0
5 -0.1 1 0
6 0.1 1 0
End Nodes
Begin Elements Interface2Plus2N
10 0 1 2 4 5
11 0 1 3 4 6
End Elements
''')
                for name, entities in (
                    ('Soil', ()), ('Left', (10,)),
                    ('LeftAlias', (10,)), ('Right', (11,))
                ):
                    output.write(f'Begin SubModelPart {name}\nBegin SubModelPartNodes\n'
                                 '1\n2\n3\n4\n5\n6\nEnd SubModelPartNodes\n')
                    if entities:
                        output.write('Begin SubModelPartElements\n')
                        output.writelines(f'{entity}\n' for entity in entities)
                        output.write('End SubModelPartElements\n')
                    output.write('End SubModelPart\n')
            with open(project_path, 'w', encoding='utf-8') as output:
                json.dump({'stages': {'RightOnly': {'solver_settings': {
                    'problem_domain_sub_model_part_list': ['Soil', 'Right']}}}}, output)

            with patch('matplotlib.pyplot.show'), patch('builtins.print'), patch(
                'visualize_mdpa_project.CheckButtons', side_effect=capture_checks
            ), patch('visualize_mdpa_project.Button', side_effect=capture_buttons):
                visualize_mdpa(mdpa_path, [project_path])
                figure = plt.gcf()
                axes = next(axis for axis in figure.axes
                            if axis.get_title().startswith('MDPA Visualization'))
                controls = figure._mdpa_diagnostic_widgets
                controls['gap_increase']._observers.process('clicked', None)
                controls['separated_faces']._observers.process('clicked', None)
                expected = 0.01 * 0.002 * 1.1 * np.array([0.1, 1]) / np.linalg.norm([0.1, 1])

                def is_shifted():
                    return any(np.allclose(point, expected, atol=1.0e-12)
                               for collection in axes.collections
                               if isinstance(collection, PathCollection)
                               for point in collection.get_offsets())

                def toggle(name):
                    widget = next(widget for widget in reversed(checks)
                                  if name in [label.get_text() for label in widget.labels])
                    index = [label.get_text() for label in widget.labels].index(name)
                    widget.set_active(index)

                self.assertFalse(is_shifted())
                toggle('LeftAlias')
                self.assertFalse(is_shifted())
                toggle('Left')
                self.assertTrue(is_shifted())
                toggle('Left')
                self.assertFalse(is_shifted())
                toggle('RightOnly')
                self.assertTrue(is_shifted())
                for label, shifted in (('Hide', False), ('Show', True),
                                       ('Invert', False), ('Invert', True)):
                    button = next(widget for widget in buttons
                                  if widget.label.get_text() == label)
                    button._observers.process('clicked', None)
                    self.assertEqual(shifted, is_shifted())
                toggle('All')
                self.assertFalse(is_shifted())
                toggle('RightOnly')
                self.assertTrue(is_shifted())
                controls['separated_faces']._observers.process('clicked', None)
                self.assertFalse(is_shifted())
                self.assertEqual((0, 0, 0), parse_mdpa_data(mdpa_path).nodes[4])

    def test_separated_face_controls_lookup_stages_and_restoration(self):
        from matplotlib.collections import PathCollection
        from matplotlib.quiver import Quiver
        from matplotlib.widgets import Button as RealButton
        from matplotlib.widgets import CheckButtons as RealCheckButtons

        with tempfile.TemporaryDirectory() as tmp_dir:
            mdpa_path = os.path.join(tmp_dir, 'faces.mdpa')
            project_path = os.path.join(tmp_dir, 'ProjectParameters.json')
            with open(mdpa_path, 'w', encoding='utf-8') as output:
                output.write('''Begin Nodes
1 0 0 0
2 2 0 0
3 1 0 0
4 0 0 0
5 2 0 0
6 1 0 0
End Nodes
Begin Elements Interface3Plus3N
10 0 1 2 3 4 5 6
End Elements
Begin SubModelPart Interface
Begin SubModelPartNodes
1
2
3
4
5
6
End SubModelPartNodes
Begin SubModelPartElements
10
End SubModelPartElements
End SubModelPart
Begin SubModelPart Source
Begin SubModelPartNodes
1
2
3
End SubModelPartNodes
End SubModelPart
Begin SubModelPart Target
Begin SubModelPartNodes
4
5
6
End SubModelPartNodes
End SubModelPart
''')
            with open(project_path, 'w', encoding='utf-8') as output:
                json.dump({'stages': {
                    'Before': {'processes': [{'process_name': 'Constraints', 'Parameters': {
                        'master_model_part_name': 'Domain.Source',
                        'slave_model_part_name': 'Domain.Target'}}]},
                    'After': {'solver_settings': {'problem_domain_sub_model_part_list': ['Interface']}}
                }}, output)
            check_widgets = []
            button_widgets = []

            def capture_checks(*args, **kwargs):
                widget = RealCheckButtons(*args, **kwargs)
                check_widgets.append(widget)
                return widget

            def capture_buttons(*args, **kwargs):
                widget = RealButton(*args, **kwargs)
                button_widgets.append(widget)
                return widget

            with patch('matplotlib.pyplot.show'), patch('builtins.print') as print_mock, patch(
                'visualize_mdpa_project.CheckButtons', side_effect=capture_checks
            ), patch(
                'visualize_mdpa_project.Button', side_effect=capture_buttons
            ):
                visualize_mdpa(mdpa_path, [project_path])
                figure = plt.gcf()
                axes = next(axis for axis in figure.axes if axis.get_title().startswith('MDPA Visualization'))
                controls = figure._mdpa_diagnostic_widgets

                def rendered_points():
                    return {tuple(point) for collection in axes.collections
                            if isinstance(collection, PathCollection) for point in collection.get_offsets()}

                original_points = rendered_points()
                initial_gap = 0.002 * 2
                self.assertEqual(initial_gap, float(controls['gap_value'].text))
                self.assertEqual('Gap -', controls['gap_decrease'].label.get_text())
                self.assertEqual('Gap +', controls['gap_increase'].label.get_text())
                for _ in range(30):
                    controls['gap_increase']._observers.process('clicked', None)
                gap = initial_gap * 1.1 ** 30
                self.assertGreater(gap, 0.02 * 2)
                self.assertAlmostEqual(float(controls['gap_value'].text),
                                       gap, places=6)
                self.assertEqual(original_points, rendered_points())
                controls['gap_decrease']._observers.process('clicked', None)
                controls['gap_increase']._observers.process('clicked', None)
                controls['separated_faces']._observers.process('clicked', None)
                shifted_points = rendered_points()
                offset = 0.01 * gap
                self.assertTrue(any(np.allclose(point, (2 - offset, 0))
                                    for point in shifted_points))
                self.assertTrue(any(line.get_linestyle() == '--' for line in axes.lines))
                self.assertEqual('auto', axes.get_aspect())
                mesh_button = next(widget for widget in button_widgets if widget.label.get_text() == 'Draw mesh')
                normals_button = next(widget for widget in button_widgets if widget.label.get_text() == 'Draw normals')
                mesh_button._observers.process('clicked', None)
                normals_button._observers.process('clicked', None)
                self.assertTrue(any(len(line.get_xdata()) == 3 and np.allclose(line.get_xdata(), [offset, 1, 2 - offset])
                                    for line in axes.lines))
                self.assertTrue(any(np.allclose(collection.get_offsets(), [[1, 0]])
                                    for collection in axes.collections if isinstance(collection, Quiver)))
                for stage_index in (1, 2):
                    stage_widget = next(widget for widget in reversed(check_widgets)
                                        if widget.labels[0].get_text() == 'All')
                    stage_widget.set_active(stage_index)
                    self.assertEqual(original_points if stage_index == 1 else shifted_points,
                                     rendered_points())
                controls['node_id'].set_val('6')
                self.assertTrue(any(label.get_text() == 'E10' for label in axes.texts))
                self.assertTrue(any(label.get_text() == '6' for label in axes.texts))
                messages = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)
                self.assertIn('Node 6 @ (1.00, 0.00)', messages)
                self.assertIn('Incident connectivity', messages)
                self.assertIn('nodes (1, 2, 3, 4, 5, 6)', messages)
                self.assertIn('[Before] master/slave candidate', messages)
                figure.canvas.draw()
                pixel_x, pixel_y = axes.transData.transform((offset, 0))
                for event_name in ('button_press_event', 'button_release_event'):
                    event = MouseEvent(event_name, figure.canvas, pixel_x, pixel_y, button=1)
                    figure.canvas.callbacks.process(event_name, event)
                self.assertIn('Node 4 @', str(print_mock.call_args.args[0]))
                controls['labels']._observers.process('clicked', None)
                self.assertEqual([], axes.texts[:])
                controls['gap_increase']._observers.process('clicked', None)
                self.assertTrue(any(np.allclose(point, (2 - offset * 1.1, 0))
                                    for point in rendered_points()))
                controls['gap_decrease']._observers.process('clicked', None)
                self.assertTrue(any(np.allclose(point, (2 - offset, 0))
                                    for point in rendered_points()))
                controls['gap_value'].set_val('2e-1')
                self.assertTrue(any(np.allclose(point, (1.998, 0))
                                    for point in rendered_points()))
                controls['gap_increase']._observers.process('clicked', None)
                self.assertAlmostEqual(float(controls['gap_value'].text), 0.22)
                applied_points = rendered_points()
                for invalid in ('', 'invalid', '-1', '0', 'nan', 'inf', '1e999'):
                    controls['gap_value'].set_val(invalid)
                    self.assertEqual(applied_points, rendered_points())
                    self.assertEqual('Gap must be a positive finite number.',
                                     print_mock.call_args.args[0])
                controls['gap_decrease']._observers.process('clicked', None)
                self.assertAlmostEqual(float(controls['gap_value'].text), 0.2)
                self.assertTrue(any(np.allclose(collection.get_offsets(), [[1, 0]])
                                    for collection in axes.collections if isinstance(collection, Quiver)))
                controls['separated_faces']._observers.process('clicked', None)
                self.assertEqual(original_points, rendered_points())
                controls['gap_value'].set_val('10')
                self.assertEqual(original_points, rendered_points())
                controls['separated_faces']._observers.process('clicked', None)
                self.assertTrue(any(np.allclose(point, (1.9, 0))
                                    for point in rendered_points()))
                controls['separated_faces']._observers.process('clicked', None)
                self.assertFalse(any(line.get_linestyle() == '--' for line in axes.lines))
                self.assertEqual('auto', axes.get_aspect())
                controls['node_id'].set_val('not-an-id')
                controls['node_id'].set_val('999')
                self.assertEqual(parse_mdpa_data(mdpa_path).nodes[6], (1, 0, 0))

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
