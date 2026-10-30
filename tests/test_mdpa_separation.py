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

    def test_separation_skips_unsupported_topology_with_node_diagnostics(self):
        nodes = {1: (0, 0, 0), 2: (0, 0, 0), 3: (1, 1, 0), 4: (2, 0, 0)}
        parts = {'Master': {1}, 'Slave': {2}}
        relations = [('Master', 'Slave', 'Global')]
        unsupported = [
            ('Elements', 'CrBeamElement2D3N', 10, (2, 3, 4)),
            ('Conditions', 'UnknownCondition2D3N', 11, (2, 3, 4)),
            ('Elements', 'UnknownElement2D3N', 12, (2, 3, 4)),
            ('Elements', 'Interface3Plus3N', 13, (2, 3, 4)),
        ]
        supported = ('Conditions', 'LineCondition2D2N', 20, (2, 4))
        for record in unsupported:
            with self.subTest(entity_type=record[1]):
                with self.assertRaises(ValueError):
                    _connection_paths(record[1], record[3])
                for records in ([record], [record, supported]):
                    parsed = ParsedMdpa(nodes, parts, [], {}, records)
                    _, _, directions, issues = _build_face_separation(parsed, relations)
                    diagnostic = next(message for message in issues[2]
                                      if 'Unsupported separation geometry' in message)
                    self.assertIn(f'{record[0]} {record[2]}', diagnostic)
                    self.assertIn(record[1], diagnostic)
                    for node_id in record[3]:
                        self.assertIn(diagnostic, issues[node_id])
                    displayed = _separated_coordinates(nodes, directions, 0.1)
                    self.assertEqual(nodes[1], displayed[1])
                    if supported in records:
                        np.testing.assert_allclose(directions[2], (1, 0))
                        self.assertNotEqual(nodes[2], displayed[2])
                    else:
                        self.assertEqual(nodes[2], displayed[2])
                        self.assertIn('No noncoincident connected edge available; node left fixed',
                                      issues[2])
        _, _, directions, issues = _build_face_separation(
            ParsedMdpa(nodes, parts, [], {}, [unsupported[0]]))
        self.assertEqual({}, directions)
        self.assertTrue(any('CrBeamElement2D3N' in message for message in issues[2]))

    def test_gap_displacement_is_scaled_to_one_hundredth(self):
        nodes = {1: (0, 0, 2), 2: (0, 0, 2), 3: (0, 0, 2)}
        directions = {2: (0.6, 0.8), 3: (1.2, 1.6)}
        displayed = _separated_coordinates(nodes, directions, 0.1)
        np.testing.assert_allclose(displayed[2], (0.0006, 0.0008, 2))
        np.testing.assert_allclose(displayed[3], (0.0012, 0.0016, 2))
        self.assertEqual(nodes[1], displayed[1])
        self.assertEqual(nodes, _separated_coordinates(nodes, directions, 0))
        self.assertEqual((0, 0, 2), nodes[2])

    def test_midsides_interpolate_rank_scaled_corner_displacements(self):
        nodes = {
            1: (0, 0, 0), 2: (2, 0, 0), 3: (1, 0, 0), 4: (-2, 0, 0),
            10: (0, 0, 0), 11: (-2, 0, 0),
            20: (0, 0, 0), 21: (2, 0, 0), 22: (1, 0, 0),
            30: (0, 1, 0), 31: (2, 1, 0),
        }
        records = [
            ('Elements', 'Interface2Plus2N', 100, (1, 4, 10, 11)),
            ('Elements', 'Interface3Plus3N', 101, (1, 2, 3, 20, 21, 22)),
            ('Elements', 'UPwSmallStrainElement2D3N', 102, (10, 11, 30)),
            ('Elements', 'SmallStrainElement2D4N', 103, (20, 21, 31, 30)),
        ]
        for ordered in (records, records[::-1]):
            with self.subTest(reversed_order=ordered is not records):
                parsed = ParsedMdpa(nodes, {}, [], {}, ordered)
                _, _, directions, issues = _build_face_separation(parsed)
                displayed = _separated_coordinates(nodes, directions, 0.1)
                np.testing.assert_allclose(directions[20], (0, 2))
                np.testing.assert_allclose(directions[21], (0, 1))
                np.testing.assert_allclose(displayed[20], (0, 0.002, 0))
                np.testing.assert_allclose(displayed[21], (2, 0.001, 0))
                np.testing.assert_allclose(displayed[22], (1, 0.0015, 0))
                np.testing.assert_allclose(
                    np.array(displayed[22]) - nodes[22],
                    ((np.array(displayed[20]) - nodes[20])
                     + (np.array(displayed[21]) - nodes[21])) / 2)
                for reference in (1, 2, 3, 4):
                    self.assertEqual(nodes[reference], displayed[reference])
                self.assertEqual(nodes, _separated_coordinates(nodes, directions, 0))
                self.assertEqual({}, issues)

    def test_face_separation_follows_connected_soil_edge(self):
        nodes = {1: (0, 0, 0), 2: (2, 0, 0),
                 3: (0, 0, 0), 4: (2, 0, 0), 5: (1, 2, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface2Plus2N', 10, (1, 2, 3, 4)),
            ('Elements', 'UPwSmallStrainElement2D3N', 11, (3, 4, 5))])
        _, _, directions, issues = _build_face_separation(parsed)
        np.testing.assert_allclose(directions[3], (1 / np.sqrt(5), 2 / np.sqrt(5)))
        np.testing.assert_allclose(directions[4], (-1 / np.sqrt(5), 2 / np.sqrt(5)))
        displayed = _separated_coordinates(nodes, directions, 0.1)
        np.testing.assert_allclose(displayed[3][:2], np.array(nodes[5][:2]) * 0.001 / np.sqrt(5))
        self.assertEqual(nodes[1], displayed[1])
        self.assertEqual({}, issues)

    def test_face_separation_orients_into_linear_quadrilateral_soil(self):
        nodes = {1: (0, 0, 0), 2: (2, 0, 0),
                 3: (0, 0, 0), 4: (2, 0, 0),
                 5: (2, -1, 0), 6: (0, -1, 0),
                 7: (2, 10, 0), 8: (0, 10, 0)}
        soil = ('Elements', 'SmallStrainElement2D4N', 11, (3, 4, 5, 6))
        condition = ('Conditions', 'SurfaceCondition2D4N', 12, (3, 4, 7, 8))
        for connectivity in ((1, 2, 3, 4), (2, 1, 4, 3)):
            for with_condition in (False, True):
                records = [('Elements', 'Interface2Plus2N', 10, connectivity), soil]
                if with_condition:
                    records.append(condition)
                for ordered in (records, records[::-1]):
                    with self.subTest(connectivity=connectivity,
                                      with_condition=with_condition,
                                      reversed_order=ordered is not records):
                        parsed = ParsedMdpa(nodes, {}, [], {}, ordered)
                        _, _, directions, issues = _build_face_separation(parsed)
                        displayed = _separated_coordinates(nodes, directions, 0.1)
                        for node_id in (3, 4):
                            np.testing.assert_allclose(directions[node_id], (0, -1))
                            np.testing.assert_allclose(
                                np.array(displayed[node_id]) - nodes[node_id],
                                (0, -0.001, 0))
                        for node_id in (1, 2):
                            self.assertEqual(nodes[node_id], displayed[node_id])
                        self.assertEqual({}, issues)

    def test_face_separation_follows_quadratic_mesh_edge(self):
        nodes = {1: (0, 0, 0), 2: (2, 0, 0), 3: (0, 0, 0),
                 4: (2, 0, 0), 5: (2, 3, 0), 6: (1, 2, 0),
                 7: (2, 1.5, 0), 8: (1, -1, 0)}
        records = [('Elements', 'Interface2Plus2N', 10, (1, 2, 3, 4)),
                   ('Elements', 'SmallStrainElement2D6N', 11, (3, 4, 5, 6, 7, 8))]
        _, _, directions, issues = _build_face_separation(ParsedMdpa(nodes, {}, [], {}, records))
        np.testing.assert_allclose(directions[3], (1 / np.sqrt(5), 2 / np.sqrt(5)))
        np.testing.assert_allclose(directions[6], (directions[3] + directions[4]) / 2)
        np.testing.assert_allclose(directions[7], directions[4] / 2)
        np.testing.assert_allclose(directions[8], directions[3] / 2)
        displayed = _separated_coordinates(nodes, directions, 0.1)
        for midside, first, second in ((6, 3, 4), (7, 4, 5), (8, 5, 3)):
            displacement = np.array(displayed[midside][:2]) - nodes[midside][:2]
            first_displacement = np.array(displayed[first][:2]) - nodes[first][:2]
            second_displacement = np.array(displayed[second][:2]) - nodes[second][:2]
            np.testing.assert_allclose(displacement, (first_displacement + second_displacement) / 2)
        _, _, reversed_directions, _ = _build_face_separation(
            ParsedMdpa(nodes, {}, [], {}, records[::-1]))
        np.testing.assert_allclose(directions[3], reversed_directions[3])
        for midside in (6, 7, 8):
            np.testing.assert_allclose(directions[midside], reversed_directions[midside])
        self.assertEqual({}, issues)

    def test_quadratic_line_and_quadrilateral_interpolate_midsides(self):
        nodes = {1: (0, 0, 0), 2: (0, 0, 0), 3: (2, 0, 0),
                 4: (2, 2, 0), 5: (0, 2, 0), 6: (1, 0, 0),
                 7: (2, 1, 0), 8: (1, 2, 0), 9: (0, 1, 0)}
        cases = [
            (('Conditions', 'LineCondition2D3N', 10, (2, 3, 6)), ((6, 2, 3),)),
            (('Elements', 'SmallStrainElement2D8N', 10, (2, 3, 4, 5, 6, 7, 8, 9)),
             ((6, 2, 3), (7, 3, 4), (8, 4, 5), (9, 5, 2))),
        ]
        for record, edges in cases:
            with self.subTest(entity_type=record[1]):
                parsed = ParsedMdpa(nodes, {'Source': {1}, 'Target': {2}}, [], {}, [record])
                _, _, directions, issues = _build_face_separation(
                    parsed, [('Source', 'Target', 'Before')], visible_interfaces=set())
                displayed = _separated_coordinates(nodes, directions, 0.1)
                for midside, first, second in edges:
                    expected = (np.array(directions.get(first, (0, 0)))
                                + directions.get(second, (0, 0))) / 2
                    np.testing.assert_allclose(directions[midside], expected)
                    np.testing.assert_allclose(
                        np.array(displayed[midside][:2]) - nodes[midside][:2], 0.001 * expected)
                self.assertEqual(nodes[1], displayed[1])
                self.assertEqual(nodes, _separated_coordinates(nodes, directions, 0))
                self.assertEqual({}, issues)

    def test_constraint_candidates_follow_mesh_edges_without_boundary_normals(self):
        nodes = {1: (0, 0, 0), 2: (0, 0, 0), 3: (1, 2, 0), 4: (2, 0, 0)}
        parsed = ParsedMdpa(nodes, {'Source': {1}, 'Target': {2}}, [], {}, [
            ('Elements', 'UPwSmallStrainElement2D3N', 1, (2, 3, 4))])
        _, _, directions, issues = _build_face_separation(parsed, [('Source', 'Target', 'Before')])
        np.testing.assert_allclose(directions[2], (1 / np.sqrt(5), 2 / np.sqrt(5)))
        self.assertEqual({}, issues)

    def test_face_separation_keeps_references_and_source_coordinates(self):
        nodes = {1: (0, 0, 0), 2: (2, 0, 0), 3: (1, 0, 0),
                 4: (0, 0, 0), 5: (2, 0, 0), 6: (1, 0, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface3Plus3N', 10, (1, 2, 3, 4, 5, 6))])
        incident, pairs, normals, issues = _build_face_separation(parsed)
        self.assertEqual({(1, 4), (2, 5), (3, 6)}, set(pairs))
        self.assertEqual(1, len(incident[6]))
        self.assertEqual({}, issues)
        displayed = _separated_coordinates(nodes, normals, 0.1)
        self.assertEqual(nodes[1], displayed[1])
        np.testing.assert_allclose(normals[6], (normals[4] + normals[5]) / 2)
        self.assertEqual((1, 0, 0), displayed[6])
        self.assertEqual((1, 0, 0), nodes[6])
        self.assertEqual(nodes, _separated_coordinates(nodes, normals, 0))

    def test_constraint_roles_override_node_ids_and_interface_defaults(self):
        nodes = {1: (0, 0, 0), 2: (2, 0, 0), 3: (1, 0, 0),
                 4: (0, 0, 0), 5: (2, 0, 0), 6: (1, 0, 0),
                 7: (0, 1, 0)}
        records = [
            ('Elements', 'Interface3Plus3N', 10, (1, 2, 3, 4, 5, 6)),
            ('Elements', 'UPwSmallStrainElement2D3N', 11, (1, 2, 7)),
        ]
        parsed = ParsedMdpa(nodes, {'Master': {4, 5, 6}, 'Slave': {1, 2, 3}},
                            [], {}, records)
        for visible in (None, set()):
            with self.subTest(visible_interfaces=visible):
                _, pairs, directions, issues = _build_face_separation(
                    parsed, [('Master', 'Slave', 'Global')], visible_interfaces=visible)
                self.assertIn((1, 4), pairs)
                displayed = _separated_coordinates(nodes, directions, 0.1)
                for master in (4, 5, 6):
                    self.assertEqual(nodes[master], displayed[master])
                self.assertNotEqual(nodes[1], displayed[1])
                self.assertNotEqual(nodes[2], displayed[2])
                if visible is None:
                    np.testing.assert_allclose(directions[3],
                                               (directions[1] + directions[2]) / 2)
                    self.assertNotEqual(nodes[3], displayed[3])
                    self.assertEqual({}, issues)
                else:
                    np.testing.assert_allclose(directions[1], (1, 0))
                    np.testing.assert_allclose(directions[2], (-1, 0))
                    self.assertNotIn(3, directions)
                    self.assertEqual(nodes[3], displayed[3])
                    self.assertEqual({
                        3: {'No noncoincident connected edge available; node left fixed'}
                    }, issues)
                self.assertEqual(nodes, _separated_coordinates(nodes, directions, 0))

    def test_hidden_interfaces_do_not_dilute_shared_midside_interpolation(self):
        nodes = {
            1: (0, 0, 0), 2: (2, 0, 0), 3: (1, 0, 0),
            4: (0, 0, 0), 5: (2, 0, 0), 6: (1, 0, 0),
            7: (2, -1, 0), 8: (0, -1, 0),
            9: (0, 0, 0), 10: (2, 0, 0),
            11: (0, 0, 0), 12: (2, 0, 0),
        }
        visible = ('Elements', 'Interface3Plus3N', 20, (1, 2, 3, 4, 5, 6))
        hidden = ('Elements', 'Interface3Plus3N', 21, (9, 10, 3, 11, 12, 6))
        soil = ('Elements', 'SmallStrainElement2D4N', 22, (4, 5, 7, 8))
        _, _, baseline, _ = _build_face_separation(
            ParsedMdpa(nodes, {}, [], {}, [visible, soil]))
        np.testing.assert_allclose(baseline[6], (0, -1))
        for records in ([visible, hidden, soil], [soil, hidden, visible]):
            with self.subTest(reversed_order=records[0] == soil):
                parsed = ParsedMdpa(nodes, {}, [], {}, records)
                _, _, directions, issues = _build_face_separation(
                    parsed, visible_interfaces={('Elements', 20)})
                np.testing.assert_allclose(directions[6], baseline[6])
                displayed = _separated_coordinates(nodes, directions, 0.1)
                np.testing.assert_allclose(displayed[6], (1, -0.001, 0))
                for node_id in (9, 10, 11, 12):
                    self.assertEqual(nodes[node_id], displayed[node_id])
                self.assertEqual({}, issues)
                _, _, directions, _ = _build_face_separation(
                    parsed, visible_interfaces=set())
                self.assertNotIn(6, directions)
                self.assertEqual(nodes, _separated_coordinates(nodes, directions, 0.1))
                _, _, implicit_all, implicit_issues = _build_face_separation(parsed)
                _, _, explicit_all, explicit_issues = _build_face_separation(
                    parsed, visible_interfaces={('Elements', 20), ('Elements', 21)})
                self.assertEqual(set(implicit_all), set(explicit_all))
                for node_id in implicit_all:
                    np.testing.assert_allclose(implicit_all[node_id], explicit_all[node_id])
                self.assertEqual(implicit_issues, explicit_issues)

    def test_ambiguous_constraint_slaves_and_conflicting_roles_stay_fixed(self):
        nodes = {1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 0, 0),
                 4: (1, 1, 0), 5: (2, 0, 0)}
        parsed = ParsedMdpa(
            nodes, {'Masters': {2, 3}, 'Slaves': {1}}, [], {},
            [('Elements', 'UPwSmallStrainElement2D3N', 10, (1, 4, 5))])
        _, _, directions, issues = _build_face_separation(
            parsed, [('Masters', 'Slaves', 'Global')])
        self.assertEqual(nodes[1], _separated_coordinates(nodes, directions, 0.1)[1])
        self.assertIn('Multiple coincident master candidates', issues[1])
        parsed.submodel_nodes = {'Masters': {2}, 'Slaves': {1}}
        _, _, directions, issues = _build_face_separation(
            parsed, [('Masters', 'Slaves', 'Global'), ('Slaves', 'Masters', 'Global')])
        displayed = _separated_coordinates(nodes, directions, 0.1)
        for node_id in (1, 2):
            self.assertEqual(nodes[node_id], displayed[node_id])
            self.assertIn('Conflicting master/slave roles; node left fixed', issues[node_id])

    def test_constraint_ambiguity_accumulates_masters_across_relations(self):
        nodes = {1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 0, 0),
                 4: (1, 1, 0), 5: (2, 0, 0)}
        parsed = ParsedMdpa(
            nodes, {'A': {2}, 'B': {3}, 'S': {1}}, [], {},
            [('Elements', 'UPwSmallStrainElement2D3N', 10, (1, 4, 5))])
        relations = [('A', 'S', 'Before'), ('B', 'S', 'After')]
        for considered in (relations, relations[::-1]):
            with self.subTest(relations=considered):
                _, pairs, directions, issues = _build_face_separation(parsed, considered)
                self.assertEqual({(1, 2), (1, 3)}, set(pairs))
                self.assertEqual({('Before', 'A', 'S')}, pairs[(1, 2)]['constraints'])
                self.assertEqual({('After', 'B', 'S')}, pairs[(1, 3)]['constraints'])
                self.assertEqual(nodes[1],
                                 _separated_coordinates(nodes, directions, 0.1)[1])
                self.assertIn('Multiple coincident master candidates', issues[1])
        for relation in relations:
            with self.subTest(remaining_relation=relation):
                _, _, directions, issues = _build_face_separation(parsed, [relation])
                self.assertNotEqual(nodes[1],
                                    _separated_coordinates(nodes, directions, 0.1)[1])
                self.assertEqual({}, issues)

    def test_overlapping_master_parts_and_repeated_relations_are_not_ambiguous(self):
        nodes = {1: (0, 0, 0), 2: (0, 0, 0), 3: (1, 1, 0), 4: (2, 0, 0)}
        parsed = ParsedMdpa(
            nodes, {'A': {2}, 'B': {2}, 'S': {1}}, [], {},
            [('Elements', 'UPwSmallStrainElement2D3N', 10, (1, 3, 4))])
        relations = [('A', 'S', 'Before'), ('B', 'S', 'After'), ('A', 'S', 'Before')]
        _, pairs, directions, issues = _build_face_separation(parsed, relations)
        self.assertEqual({(1, 2)}, set(pairs))
        self.assertEqual({('Before', 'A', 'S'), ('After', 'B', 'S')},
                         pairs[(1, 2)]['constraints'])
        displayed = _separated_coordinates(nodes, directions, 0.1)
        self.assertNotEqual(nodes[1], displayed[1])
        self.assertEqual(nodes[2], displayed[2])
        self.assertEqual({}, issues)

    def test_face_separation_combines_compatible_interface_directions(self):
        nodes = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0),
                 4: (0, 0, 0), 5: (1, 0, 0), 6: (1, 1, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface2Plus2N', 10, (1, 2, 4, 5)),
            ('Elements', 'Interface2Plus2N', 11, (2, 3, 5, 6))])
        _, _, normals, issues = _build_face_separation(parsed)
        np.testing.assert_allclose(normals[5], (-0.5, 0.5))
        self.assertEqual({}, issues)

    def test_face_separation_inclined_and_reversed_interfaces(self):
        nodes = {1: (0, 0, 0), 2: (1, 1, 0),
                 3: (0, 0, 0), 4: (1, 1, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface2Plus2N', 10, (1, 2, 3, 4)),
            ('Elements', 'Interface2Plus2N', 11, (2, 1, 4, 3))])
        _, _, normals, issues = _build_face_separation(parsed)
        np.testing.assert_allclose(normals[3], (1 / np.sqrt(2), 1 / np.sqrt(2)))
        self.assertEqual({}, issues)

    def test_constraint_candidates_without_connected_edges_are_not_moved(self):
        nodes = {1: (0, 0, 0), 2: (0, 0, 0), 3: (1, 0, 0)}
        parsed = ParsedMdpa(nodes, {'masters': {1}, 'slaves': {2, 3}}, [], {}, [])
        _, pairs, normals, issues = _build_face_separation(
            parsed, [('masters', 'slaves', 'stage_1')])
        self.assertEqual({(1, 2)}, set(pairs))
        self.assertEqual({('stage_1', 'masters', 'slaves')}, pairs[(1, 2)]['constraints'])
        self.assertEqual({}, normals)
        self.assertIn('No noncoincident connected edge available; node left fixed', issues[2])

    def test_qualified_constraint_names_resolve_before_pairing(self):
        nodes = {1: (0, 0, 0), 10: (0, 0, 0), 2: (1, 2, 0), 3: (2, 0, 0)}
        parsed = ParsedMdpa(
            nodes, {'Structure.Wall': {10}, 'Soil.Face': {1},
                    'Wall': {2}, 'Face': {3}, 'Other.Wall': {3}}, [], {},
            [('Elements', 'UPwSmallStrainElement2D3N', 1, (1, 2, 3))])
        relations = [('Domain.Structure.Wall', 'Domain.Soil.Face', 'Before')]
        resolved = _resolve_constraint_relations(relations, parsed.submodel_nodes)
        self.assertEqual([('Structure.Wall', 'Soil.Face', 'Before')], resolved)
        self.assertEqual(resolved, _resolve_constraint_relations(
            resolved, parsed.submodel_nodes))
        _, pairs, directions, issues = _build_face_separation(parsed, relations)
        self.assertEqual({(1, 10)}, set(pairs))
        self.assertEqual({('Before', 'Structure.Wall', 'Soil.Face')},
                         pairs[(1, 10)]['constraints'])
        displayed = _separated_coordinates(nodes, directions, 0.1)
        self.assertEqual(nodes[10], displayed[10])
        self.assertNotEqual(nodes[1], displayed[1])
        self.assertEqual({}, issues)
        for hidden in ({'Structure.Wall'}, {'Soil.Face'}):
            visible = [relation for relation in resolved
                       if relation[0] not in hidden and relation[1] not in hidden]
            _, pairs, directions, _ = _build_face_separation(parsed, visible)
            self.assertEqual({}, pairs)
            self.assertEqual(nodes, _separated_coordinates(nodes, directions, 0.1))
        with patch('builtins.print') as print_mock:
            unresolved = _resolve_constraint_relations(
                [('Domain.Missing.Wall', 'Domain.Soil.Face', 'Before')],
                {'Structure.Wall': {10}, 'Soil.Face': {1}, 'Other.Wall': {3}})
        self.assertEqual([], unresolved)
        self.assertIn('Domain.Missing.Wall', print_mock.call_args.args[0])

    def test_face_separation_multiple_clones_have_distinct_positions(self):
        nodes = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 0, 0),
                 4: (1, 0, 0), 5: (0, 0, 0), 6: (1, 0, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface2Plus2N', 10, (1, 2, 3, 4)),
            ('Elements', 'Interface2Plus2N', 11, (3, 4, 5, 6))])
        _, _, normals, _ = _build_face_separation(parsed)
        displayed = _separated_coordinates(nodes, normals, 0.1)
        self.assertEqual(nodes[1], displayed[1])
        np.testing.assert_allclose(displayed[3], (0.001, 0, 0))
        np.testing.assert_allclose(displayed[5], (0.002, 0, 0))

    def test_face_separation_quadratic_curve_averages_corner_vectors(self):
        nodes = {1: (0, 0, 0), 2: (2, 0, 0), 3: (1, 0.5, 0),
                 4: (0, 0, 0), 5: (2, 0, 0), 6: (1, 0.5, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface3Plus3N', 10, (1, 2, 3, 4, 5, 6))])
        _, _, normals, _ = _build_face_separation(parsed)
        np.testing.assert_allclose(normals[4], (2 / np.sqrt(5), 1 / np.sqrt(5)))
        np.testing.assert_allclose(normals[5], (-2 / np.sqrt(5), 1 / np.sqrt(5)))
        np.testing.assert_allclose(normals[6], (normals[4] + normals[5]) / 2)
        np.testing.assert_allclose(normals[6], (0, 1 / np.sqrt(5)))

    def test_face_separation_conflicting_normals_leave_node_fixed(self):
        nodes = {1: (0, 0, 0), 2: (-0.1, 1, 0), 3: (0.1, 1, 0),
                 4: (0, 0, 0), 5: (-0.1, 1, 0), 6: (0.1, 1, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface2Plus2N', 10, (1, 2, 4, 5)),
            ('Elements', 'Interface2Plus2N', 11, (1, 3, 4, 6))])
        _, _, normals, issues = _build_face_separation(parsed)
        np.testing.assert_allclose(normals[4], (0, 0))
        self.assertIn('Conflicting visible interface directions; node left fixed', issues[4])
        self.assertEqual(nodes[4], _separated_coordinates(nodes, normals, 0.1)[4])
        for entity_id, neighbor in ((10, 5), (11, 6)):
            with self.subTest(visible_interface=entity_id):
                _, _, directions, diagnostics = _build_face_separation(
                    parsed, visible_interfaces={('Elements', entity_id)})
                expected = np.array(nodes[neighbor][:2]) / np.linalg.norm(nodes[neighbor][:2])
                np.testing.assert_allclose(directions[4], expected)
                self.assertEqual({}, diagnostics)
        _, pairs, directions, issues = _build_face_separation(
            parsed, visible_interfaces=set())
        self.assertEqual({}, pairs)
        self.assertEqual({}, directions)
        self.assertEqual({}, issues)
        _, _, reversed_directions, reversed_issues = _build_face_separation(
            ParsedMdpa(nodes, {}, [], {}, parsed.entity_records[::-1]))
        np.testing.assert_allclose(reversed_directions[4], (0, 0))
        self.assertIn('Conflicting visible interface directions; node left fixed',
                      reversed_issues[4])

    def test_face_separation_opposite_normals_leave_corner_fixed(self):
        nodes = {1: (0, 0, 0), 2: (4, 0, 0), 3: (1, 1, 0),
                 4: (0, 0, 0), 5: (4, 0, 0), 6: (1, 1, 0),
                 7: (1, -1, 0), 8: (1, -1, 0)}
        parsed = ParsedMdpa(nodes, {}, [], {}, [
            ('Elements', 'Interface3Plus3N', 10, (1, 2, 3, 4, 5, 6)),
            ('Elements', 'Interface3Plus3N', 11, (1, 2, 7, 4, 5, 8))])
        _, _, directions, issues = _build_face_separation(parsed)
        np.testing.assert_allclose(directions[4], (0, 0))
        self.assertIn('Conflicting visible interface directions; node left fixed', issues[4])
        self.assertEqual(nodes[4], _separated_coordinates(nodes, directions, 0.1)[4])

    def test_shared_tip_uses_soil_centers_from_each_interface_face(self):
        nodes = {
            27: (18, -12.5, 0), 1241: (18, -12.45, 0),
            1270: (18, -12.475, 0), 21332: (18, -12.5, 0),
            21333: (18, -12.45, 0), 21362: (18, -12.475, 0),
            21692: (18, -12.45, 0), 21721: (18, -12.475, 0),
            30001: (17, -12.6, 0), 30002: (20, -12.6, 0),
        }
        records = [
            ('Elements', 'Interface3Plus3N', 6223,
             (1241, 27, 1270, 21333, 21332, 21362)),
            ('Elements', 'Interface3Plus3N', 6403,
             (27, 1241, 1270, 21332, 21692, 21721)),
            ('Elements', 'UPwSmallStrainElement2D3N', 7000,
             (21332, 21333, 30001)),
            ('Elements', 'UPwSmallStrainElement2D3N', 7001,
             (21332, 21692, 30002)),
        ]
        parsed = ParsedMdpa(nodes, {}, [], {}, records)
        for ordered_records in (records, records[::-1]):
            with self.subTest(reversed_order=ordered_records is not records):
                _, _, directions, issues = _build_face_separation(
                    ParsedMdpa(nodes, {}, [], {}, ordered_records))
                np.testing.assert_allclose(directions[21332], (0, 0))
                self.assertEqual(nodes[21332],
                                 _separated_coordinates(nodes, directions, 0.1)[21332])
                self.assertIn('Conflicting visible interface directions; node left fixed',
                              issues[21332])
        for entity_id, horizontal_sign in ((6223, -1), (6403, 1)):
            with self.subTest(visible_interface=entity_id):
                _, _, directions, issues = _build_face_separation(
                    parsed, visible_interfaces={('Elements', entity_id)})
                self.assertEqual(horizontal_sign, np.sign(directions[21332][0]))
                self.assertNotIn(21332, issues)
        self.assertEqual((18, -12.5, 0), nodes[21332])

    def test_eemdijk_shared_tip_stays_fixed_when_both_interfaces_are_visible(self):
        mdpa_path = os.path.join(
            os.path.dirname(__file__), 'fixtures', 'eemdijk_shared_tip.mdpa')
        parsed = parse_mdpa_data(mdpa_path)
        _, pairs, directions, issues = _build_face_separation(parsed)
        self.assertEqual({('Elements', 6223), ('Elements', 6403)},
                         pairs[(27, 21332)]['interfaces'])
        np.testing.assert_allclose(directions[21332], (0, 0))
        self.assertIn('Conflicting visible interface directions; node left fixed',
                      issues[21332])
        self.assertEqual(parsed.nodes[21332],
                         _separated_coordinates(parsed.nodes, directions, 0.1)[21332])

    def test_constraint_candidates_use_tolerance_and_boundary_edges(self):
        nodes = {1: (0, 0, 0), 2: (0, 1.0e-9, 0), 3: (1, 1.0e-9, 0)}
        parsed = ParsedMdpa(nodes, {'Source': {1}, 'Target': {2}}, [], {}, [
            ('Conditions', 'LineCondition2D2N', 1, (2, 3))])
        _, pairs, normals, issues = _build_face_separation(parsed, [('Source', 'Target', 'Before')])
        self.assertEqual({(1, 2)}, set(pairs))
        np.testing.assert_allclose(normals[2], (1, 0))
        self.assertEqual({}, issues)
