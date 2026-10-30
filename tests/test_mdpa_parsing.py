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
    Begin SubModelPartTables
    End SubModelPartTables
    Begin SubModelPartData
    End SubModelPartData
    Begin SubModelPartProperties
        1
    End SubModelPartProperties
  Begin SubModelPartNodes
    1
    2
  End SubModelPartNodes
End SubModelPart

Begin SubModelPart Structure.Wall
    Begin SubModelPartTables
    End SubModelPartTables
    Begin SubModelPartProperties
        2
    End SubModelPartProperties
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

    def test_kratos_connection_paths_close_four_node_quadrilateral(self):
        self.assertEqual(
            [(11, 22, 33, 44, 11)],
            _connection_paths("SmallStrainUPwElement2D4N", (11, 22, 33, 44)),
        )

    def test_connection_paths_use_explicit_three_node_geometries(self):
        cases = {
            "LinearTimoshenkoCurvedBeamElement2D3N": [(1, 3, 2)],
            "LineLoadDiffOrderCondition2D3N": [(1, 3, 2)],
            "LineNormalLoadDiffOrderCondition2D3N": [(1, 3, 2)],
            "LineCondition2D3N": [(1, 3, 2)],
            "TransientPwElement2D3N": [(1, 2, 3, 1)],
            "SteadyStatePwElement2D3N": [(1, 2, 3, 1)],
            "TransientPwLineElement2D3N": [(1, 3, 2)],
            "UPwSmallStrainElement2D3N": [(1, 2, 3, 1)],
            "UPwSmallStrainFICElement2D3N": [(1, 2, 3, 1)],
            "UPwSmallStrainAxisymmetricElement2D3N": [(1, 2, 3, 1)],
            "UPwSmallStrainAxisymmetricFICElement2D3N": [(1, 2, 3, 1)],
            "UPwUpdatedLagrangianElement2D3N": [(1, 2, 3, 1)],
            "UPwUpdatedLagrangianFICElement2D3N": [(1, 2, 3, 1)],
            "UPwUpdatedLagrangianAxisymmetricElement2D3N": [(1, 2, 3, 1)],
            "UPwUpdatedLagrangianAxisymmetricFICElement2D3N": [(1, 2, 3, 1)],
            "GeoTransientThermalElement2D3N": [(1, 2, 3, 1)],
            "GeoTransientThermalLineElement2D3N": [(1, 3, 2)],
            "Element2D3N": [(1, 2, 3, 1)],
        }
        for entity_type, expected in cases.items():
            for spelling in (entity_type, entity_type.lower()):
                with self.subTest(entity_type=spelling):
                    self.assertEqual(expected, _connection_paths(spelling, (1, 2, 3)))

    def test_connection_paths_reject_unknown_three_node_geometry(self):
        for entity_type in ("UnknownElement2D3N", "UnknownBeamElement2D3N"):
            with self.subTest(entity_type=entity_type):
                with self.assertRaisesRegex(ValueError, "Unknown geometry"):
                    _connection_paths(entity_type, (1, 2, 3))

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
