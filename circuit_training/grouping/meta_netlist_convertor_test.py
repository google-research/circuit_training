# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for meta_netlist_convertor."""
import tensorflow as tf
from absl.testing import parameterized
from circuit_training.grouping import meta_netlist_convertor
from circuit_training.grouping import meta_netlist_data_structure as mnds
from google.protobuf import text_format

from circuit_training.utils import test_utils

_TEST_NDOE_DEF_PORT = """
name: "P0"
input: "P0_M0"
input: "S0"
attr {
  key: "type"
  value {
    placeholder: "port"
  }
}
attr {
  key: "side"
  value {
    placeholder: "left"
  }
}
attr {
  key: "empty"
}
"""

_TEST_NODE_DEF_STDCELL = """
name: "S0"
input: "S1"
attr {
  key: "type"
  value {
    placeholder: "stdcell"
  }
}
attr {
  key: "ref_name"
  value {
    placeholder: "X"
  }
}
attr {
  key: "width"
  value {
    f: 2.208000
  }
}
attr {
  key: "height"
  value {
    f: 0.480000
  }
}
"""

_TEST_NODE_DEF_MACRO = """
name: "Grp_2/Pinput"
attr {
  key: "type"
  value {
    placeholder: "macro"
  }
}
attr {
  key: "width"
  value {
    f: 120
  }
}
attr {
  key: "height"
  value {
    f: 120
  }
}
"""

_TEST_NODE_DEF_MACRO_ORIE = """
name: "M0"
attr {
  key: "orientation"
  value {
    placeholder: "N"
  }
}
attr {
  key: "type"
  value {
    placeholder: "MACRO"
  }
}
attr {
  key: "width"
  value {
    f: 120
  }
}
attr {
  key: "height"
  value {
    f: 120
  }
}
attr {
  key: "x"
  value {
    f: 20
  }
}
attr {
  key: "y"
  value {
    f: 60
  }
}
"""

_TEST_NODE_DEF_MACRO_PIN = """
name: "P0_M0"
attr {
  key: "macro_name"
  value {
    placeholder: "M0"
  }
}
attr {
  key: "type"
  value {
    placeholder: "MACRO_PIN"
  }
}
attr {
  key: "x"
  value {
    f: -20
  }
}
attr {
  key: "y"
  value {
    f: 60
  }
}
attr {
  key: "x_offset"
  value {
    f: 50
  }
}
attr {
  key: "y_offset"
  value {
    f: 70
  }
}
"""

_NETLIST_FILE_PATH = "third_party/py/circuit_training/grouping/testdata/simple.pb.txt"
_ONE_NODE_GRAPH_FILE_PATH = "third_party/py/circuit_training/grouping/testdata/one_node_graph.pb.txt"


class MetaNetlistConvertorTest(parameterized.TestCase, test_utils.TestCase):

  def test_read_netlist(self):
    meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)
    self.assertLen(meta_netlist.node, 10)

  def test_read_netlist_separate(self):
    meta_netlist = meta_netlist_convertor.read_netlist(",".join(
        [_NETLIST_FILE_PATH, _ONE_NODE_GRAPH_FILE_PATH]))
    self.assertLen(meta_netlist.node, 11)

  def test_empty_netlist_raises_value_error(self):
    with self.assertRaises(ValueError):
      _ = meta_netlist_convertor.read_netlist("")

    with self.assertRaises(ValueError):
      _ = meta_netlist_convertor.read_netlist(",")

  def test_read_attr(self):
    node = text_format.Parse(_TEST_NDOE_DEF_PORT, tf.compat.v1.NodeDef())
    node_type = meta_netlist_convertor.read_attr(node, "type")
    self.assertEqual(node_type, "port")

    side = meta_netlist_convertor.read_attr(node, "side")
    self.assertEqual(side, "left")

    empty = meta_netlist_convertor.read_attr(node, "empty")
    self.assertIsNone(empty)

    not_exist = meta_netlist_convertor.read_attr(node, "not_exist")
    self.assertIsNone(not_exist)

  def test_translate_node_port(self):
    node = text_format.Parse(_TEST_NDOE_DEF_PORT, tf.compat.v1.NodeDef())
    name_to_id_map = {"P0": 0, "P0_M0": 1, "S0": 2}
    netlist = meta_netlist_convertor.translate_node(node, name_to_id_map)
    self.assertEqual(netlist.id, 0)
    self.assertEqual(netlist.type, mnds.Type.PORT)
    self.assertEqual(netlist.output_indices, [1, 2])

    node.name = "random"
    # The name won't find in the name_to_id_map.
    with self.assertRaises(KeyError):
      _ = meta_netlist_convertor.translate_node(node, name_to_id_map)

    node.name = "P0"
    # The input won't find in the name_to_id_map.
    node.input[0] = "random"
    with self.assertRaises(KeyError):
      _ = meta_netlist_convertor.translate_node(node, name_to_id_map)

    node.input[0] = "P0_M0"
    # Type is raondom, which is not defined.
    node.attr["type"].placeholder = "random"
    with self.assertRaises(KeyError):
      _ = meta_netlist_convertor.translate_node(node, name_to_id_map)

  def test_translate_node_std(self):
    node = text_format.Parse(_TEST_NODE_DEF_STDCELL, tf.compat.v1.NodeDef())
    name_to_id_map = {"S0": 0, "S1": 1}
    netlist = meta_netlist_convertor.translate_node(node, name_to_id_map)
    self.assertAlmostEqual(netlist.dimension.width, 2.208, places=5)
    self.assertAlmostEqual(netlist.dimension.height, 0.48, places=5)

  def test_translate_node_macro(self):
    node = text_format.Parse(_TEST_NODE_DEF_MACRO, tf.compat.v1.NodeDef())
    name_to_id_map = {"Grp_2/Pinput": 0}
    netlist = meta_netlist_convertor.translate_node(node, name_to_id_map)
    self.assertEqual(netlist.id, 0)
    self.assertAlmostEqual(netlist.dimension.width, 120, places=5)
    self.assertAlmostEqual(netlist.dimension.height, 120, places=5)
    self.assertTrue(netlist.soft_macro)

    # If the name doesn't tart with Grp_, the soft_macro is false.
    node = text_format.Parse(_TEST_NODE_DEF_MACRO, tf.compat.v1.NodeDef())
    node.name = "M2"
    name_to_id_map = {"M2": 0}
    netlist = meta_netlist_convertor.translate_node(node, name_to_id_map)
    self.assertFalse(netlist.soft_macro)

  def test_translate_node_macro_with_orientation(self):
    node = text_format.Parse(_TEST_NODE_DEF_MACRO_ORIE, tf.compat.v1.NodeDef())
    name_to_id_map = {"M0": 0}
    netlist = meta_netlist_convertor.translate_node(node, name_to_id_map)
    self.assertEqual(netlist.id, 0)
    self.assertAlmostEqual(netlist.orientation, mnds.Orientation.N)
    self.assertAlmostEqual(netlist.dimension.width, 120, places=5)
    self.assertAlmostEqual(netlist.dimension.height, 120, places=5)

    node.attr["type"].placeholder = "port"
    # Changing type to port raises error because orientation only exists in
    # macro node.
    with self.assertRaises(ValueError):
      _ = meta_netlist_convertor.translate_node(node, name_to_id_map)

  def test_translate_node_macro_pin(self):
    node = text_format.Parse(_TEST_NODE_DEF_MACRO_PIN, tf.compat.v1.NodeDef())
    name_to_id_map = {"P0_M0": 0, "M0": 1}
    netlist = meta_netlist_convertor.translate_node(node, name_to_id_map)
    self.assertEqual(netlist.id, 0)
    self.assertEqual(netlist.coord.x, -20)
    self.assertEqual(netlist.coord.y, 60)
    self.assertEqual(netlist.offset.x, 50)
    self.assertEqual(netlist.offset.y, 70)

  @parameterized.parameters((10, 4.08248), (0, 1e-3))
  def test_generate_canvas(self, area, side):
    canvas = meta_netlist_convertor.generate_canvas(area)
    self.assertAlmostEqual(canvas.dimension.width, side, places=5)
    self.assertAlmostEqual(canvas.dimension.height, side, places=5)

  def test_place_macro_pin(self):
    name_to_id_map = {"P0_M0": 0, "M0": 1}
    # pylint:disable=g-long-lambda.
    read_and_convert_node = lambda x: meta_netlist_convertor.translate_node(
        text_format.Parse(x, tf.compat.v1.NodeDef()), name_to_id_map)
    node_macro_pin = read_and_convert_node(_TEST_NODE_DEF_MACRO_PIN)
    node_macro = read_and_convert_node(_TEST_NODE_DEF_MACRO_ORIE)

    self.assertAlmostEqual(node_macro_pin.coord.x, -20)
    self.assertAlmostEqual(node_macro_pin.coord.y, 60)
    meta_netlist_convertor.place_macro_pin(node_macro_pin, node_macro)
    self.assertAlmostEqual(node_macro_pin.coord.x, 70)
    self.assertAlmostEqual(node_macro_pin.coord.y, 130)

  def test_convert_netlist_tf_graph_to_meta_netlist(self):
    node1 = text_format.Parse(_TEST_NODE_DEF_MACRO_PIN, tf.compat.v1.NodeDef())
    node2 = text_format.Parse(_TEST_NODE_DEF_MACRO_ORIE, tf.compat.v1.NodeDef())
    netlist_tf_graph = tf.compat.v1.MetaGraphDef()
    netlist_tf_graph.graph_def.node.append(node1)
    netlist_tf_graph.graph_def.node.append(node2)
    meta_netlist = meta_netlist_convertor.convert_tfgraph_to_meta_netlist(
        netlist_tf_graph)

    self.assertLen(meta_netlist.node, 2)
    self.assertEqual(meta_netlist.node[0].id, 0)
    self.assertEqual(meta_netlist.node[0].type, mnds.Type.MACRO_PIN)
    self.assertEqual(meta_netlist.node[0].coord.x, 70)
    self.assertEqual(meta_netlist.node[0].coord.y, 130)
    self.assertEqual(meta_netlist.node[0].offset.x, 50)
    self.assertEqual(meta_netlist.node[0].offset.y, 70)
    self.assertEqual(meta_netlist.node[0].ref_node_id, 1)

    self.assertEqual(meta_netlist.node[1].id, 1)
    self.assertEqual(meta_netlist.node[1].type, mnds.Type.MACRO)
    self.assertEqual(meta_netlist.node[1].dimension.width, 120)
    self.assertEqual(meta_netlist.node[1].dimension.height, 120)
    self.assertEqual(meta_netlist.node[1].orientation, mnds.Orientation.N)
    self.assertEqual(meta_netlist.node[1].coord.x, 20)
    self.assertEqual(meta_netlist.node[1].coord.y, 60)
    self.assertEqual(meta_netlist.node[1].input_indices[0], 0)

    self.assertAlmostEqual(
        meta_netlist.canvas.dimension.width, 154.91933, places=5)
    self.assertAlmostEqual(
        meta_netlist.canvas.dimension.height, 154.91933, places=5)
    self.assertAlmostEqual(meta_netlist.canvas.num_rows, 10)
    self.assertAlmostEqual(meta_netlist.canvas.num_columns, 10)
    self.assertAlmostEqual(meta_netlist.total_area, 14400.0)

if __name__ == "__main__":
  test_utils.main()
