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
"""Tests for grouping."""
import copy
import itertools
import os

from absl import flags
from absl.testing import absltest
import tensorflow as tf

from circuit_training.grouping import grouping
from circuit_training.grouping import meta_netlist_convertor
from circuit_training.grouping import meta_netlist_data_structure as mnds
from tensorflow.python.util.protobuf import compare
from google.protobuf import text_format
# Internal gfile dependencies


FLAGS = flags.FLAGS

_NETLIST_FILE_PATH = "third_party/py/circuit_training/grouping/testdata/simple.pb.txt"
_EXPECTED_GROUPED_NETLIST_FILE_PATH = "third_party/py/circuit_training/grouping/testdata/simple_grouped_soft_macro_not_bloated.pb.txt"
_EXPECTED_GROUPED_NETLIST_S_FILE_PATH = "third_party/py/circuit_training/grouping/testdata/simple_grouped_soft_macro_not_bloated_s.pb.txt"


class GroupingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)

  def _unplace_node_helper(self, meta_netlist: mnds.MetaNetlist) -> None:
    """Helper function for removing the coord of nodes."""
    for node in meta_netlist.node:
      if node.coord is not None and node.type != mnds.Type.MACRO_PIN:
        node.coord = None
        if node.type == mnds.Type.MACRO:
          for cind in itertools.chain(node.input_indices, node.output_indices):
            meta_netlist.node[cind].coord = None

  def test_metis_file_gen(self):
    group = grouping.Grouping(self._meta_netlist)
    output_file_path = os.path.join(FLAGS.test_tmpdir, "metis.inp")
    group.write_metis_file(output_file_path)

    with open(output_file_path, "r") as f:
      file_content = f.read()

    expected_content = "5 10\n1 7 3\n3 4\n4 9\n8 4\n10 2\n"
    self.assertEqual(file_content, expected_content)

  def test_metis_fix_file_gen(self):
    group = grouping.Grouping(self._meta_netlist)
    output_file_path = os.path.join(FLAGS.test_tmpdir, "metis.inp")
    group.setup_fixed_groups(0)
    self.assertEqual(group.num_groups(), 2)
    group.write_metis_fix_file(output_file_path)

    with open(output_file_path, "r") as f:
      file_content = f.read()

    expected_content = "-1\n-1\n-1\n-1\n-1\n-1\n0\n0\n1\n1\n"
    self.assertEqual(file_content, expected_content)

  def test_spread_metric(self):
    meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)
    group = grouping.Grouping(self._meta_netlist)
    group.setup_fixed_groups(0)
    # SpeadMetric only applies to STDCELL. There is no STDCELL in the
    # simple.pb.txt
    self.assertEqual(group.spread_metric(0), 0)

  def test_grouping(self):
    meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)
    group = grouping.Grouping(self._meta_netlist)
    group.setup_fixed_groups(0)
    self.assertEqual(group.num_groups(), 2)

    group.reset_groups()
    self.assertEqual(group.num_groups(), 0)

  def test_port_place_group_ungroup_sequences(self):
    meta_netlist = copy.deepcopy(self._meta_netlist)
    meta_netlist.canvas.dimension.width = 100
    meta_netlist.canvas.dimension.height = 100
    meta_netlist.canvas.num_columns = 10
    meta_netlist.canvas.num_rows = 10
    name_to_id_map = {node.name: node.id for node in meta_netlist.node}
    p0_id = name_to_id_map["P0"]
    p1_id = name_to_id_map["P1"]

    meta_netlist.node[p0_id].coord = mnds.Coord(x=10, y=0)
    meta_netlist.node[p1_id].coord = mnds.Coord(x=0, y=10)

    group = grouping.Grouping(meta_netlist)
    group.setup_fixed_groups(0)
    self.assertEqual(group.num_groups(), 4)

    # Make sure the ports are in expected groups.
    self.assertEqual(group.get_node_group(p0_id), 2)
    self.assertEqual(group.get_node_group(p1_id), 3)

    # Ungroup before calling SetupFixedGroups.
    group.ungroup_node(p0_id)
    group.ungroup_node(p1_id)

    # Empty groups should be deleted, ungrouping ports should reduce number of
    # groups.
    self.assertEqual(group.num_groups(), 2)
    # Put them in close proximity on the same side so they are grouped together.
    meta_netlist.node[p0_id].coord = mnds.Coord(x=0, y=19)
    meta_netlist.node[p1_id].coord = mnds.Coord(x=0, y=15)
    group.setup_fixed_groups(0)
    self.assertEqual(group.num_groups(), 3)
    # Now both ports are in the same group.
    self.assertEqual(group.get_node_group(p0_id), 2)
    self.assertEqual(group.get_node_group(p1_id), 2)

    group.ungroup_node(p0_id)
    self.assertEqual(group.num_groups(), 3)
    group.ungroup_node(p1_id)
    self.assertEqual(group.num_groups(), 2)

    # Put them far on the same side to put them into different groups.
    meta_netlist.node[p0_id].coord = mnds.Coord(x=35, y=0)
    meta_netlist.node[p1_id].coord = mnds.Coord(x=10, y=0)
    group.setup_fixed_groups(0)
    self.assertEqual(group.num_groups(), 4)
    self.assertEqual(group.get_node_group(p0_id), 3)
    self.assertEqual(group.get_node_group(p1_id), 2)

    # Test grouping connected stdcells. Traversal depth is set to 1, so the
    # first layer of stdcells within fanouts or fanins of the already grouped
    # nodes will be grouped, as well.
    group.setup_fixed_groups(1)
    s0_id = name_to_id_map["S0"]
    s1_id = name_to_id_map["S1"]

    m0_group = group.get_node_group(name_to_id_map["P0_M0"])
    m1_group = group.get_node_group(name_to_id_map["P0_M1"])
    self.assertTrue((m0_group == 0 and m1_group == 1) or
                    (m0_group == 1 and m1_group == 0))

    # S0 is driven by the port P0.
    self.assertEqual(group.get_node_group(s0_id), group.get_node_group(p0_id))
    # S1 can either be grouped in m0 or m1.
    s1_group = group.get_node_group(s1_id)
    self.assertTrue(s1_group == m0_group or s1_group == m1_group)

    # Testing resetting the groups.
    # Unplace ports and reset groups.
    self._unplace_node_helper(meta_netlist)
    group.reset_groups()
    group.setup_fixed_groups(1)
    self.assertEqual(group.get_node_group(s0_id), -1)
    s1_group = group.get_node_group(s1_id)
    self.assertTrue(s1_group == m0_group or s1_group == m1_group)

    # Increase layers of stdcells to traverse.
    group.reset_groups()
    group.setup_fixed_groups(2)
    s0_group = group.get_node_group(s0_id)
    s1_group = group.get_node_group(s1_id)
    self.assertTrue(s1_group == m0_group or s1_group == m1_group)
    # Now bot stdcells shoulde be in the same group.
    self.assertEqual(s0_group, s1_group)

  def test_write_grouped_netlist(self):
    meta_netlist = copy.deepcopy(self._meta_netlist)
    group = grouping.Grouping(meta_netlist)
    group.set_cell_area_utilization(1.0)
    name_to_id_map = {node.name: node.id for node in meta_netlist.node}
    s0_id = name_to_id_map["S0"]
    s1_id = name_to_id_map["S1"]
    group.set_node_group(s0_id, 2)
    group.set_node_group(s1_id, 2)
    # place them to check coord calculation for the group.
    meta_netlist.node[s0_id].coord = mnds.Coord(x=10, y=60)
    meta_netlist.node[s1_id].coord = mnds.Coord(x=30, y=30)
    tmpfile = os.path.join(FLAGS.test_tmpdir, "netlist.pb.txt")
    group.write_grouped_netlist(tmpfile)
    # Compare two protobufs with proto util.
    expected_graph_def = tf.compat.v1.GraphDef()

    with open(tmpfile, "r") as f:
      tmp_graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())

    expected_grouped_netlist_file_path = _EXPECTED_GROUPED_NETLIST_FILE_PATH
    with open(expected_grouped_netlist_file_path, "r") as f:
      expected_graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())

    compare.assertProto2Equal(self, tmp_graph_def, expected_graph_def)

  def test_write_grouped_netlist_with_orientation_change(self):
    meta_netlist = copy.deepcopy(self._meta_netlist)
    group = grouping.Grouping(meta_netlist)
    group.set_cell_area_utilization(1.0)
    name_to_id_map = {node.name: node.id for node in meta_netlist.node}
    s0_id = name_to_id_map["S0"]
    s1_id = name_to_id_map["S1"]
    group.set_node_group(s0_id, 2)
    group.set_node_group(s1_id, 2)
    # place them to check coord calculation for the group.
    meta_netlist.node[s0_id].coord = mnds.Coord(x=10, y=60)
    meta_netlist.node[s1_id].coord = mnds.Coord(x=30, y=30)

    meta_netlist.node[name_to_id_map["M0"]].orientation = mnds.Orientation.S

    tmpfile = os.path.join(FLAGS.test_tmpdir, "netlist.pb.txt")
    group.write_grouped_netlist(tmpfile)
    # Compare two protobufs with proto util.
    expected_graph_def = tf.compat.v1.GraphDef()

    with open(tmpfile, "r") as f:
      tmp_graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())

    expected_grouped_netlist_file_path = _EXPECTED_GROUPED_NETLIST_S_FILE_PATH
    with open(expected_grouped_netlist_file_path, "r") as f:
      expected_graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())

    compare.assertProto2Equal(self, tmp_graph_def, expected_graph_def)

  def test_merge_groups(self):
    meta_netlist = copy.deepcopy(self._meta_netlist)
    group = grouping.Grouping(meta_netlist)
    name_to_id_map = {node.name: node.id for node in meta_netlist.node}
    s0_id = name_to_id_map["S0"]
    s1_id = name_to_id_map["S1"]
    group.set_node_group(s0_id, 5)
    group.set_node_group(s1_id, 2)

    # Places them to check closeness in merge.
    meta_netlist.node[s0_id].coord = mnds.Coord(x=100, y=0)
    meta_netlist.node[s1_id].coord = mnds.Coord(x=0, y=100)

    # The distance between the two groups is 100, so the next merge should not
    # do anything.
    self.assertTrue(group.merge_small_adj_close_groups(5, 50))
    self.assertEqual(group.num_groups(), 2)

    # This time they should be merged. The function will return false, because
    # the merged group size is still smaller than 5.
    self.assertFalse(group.merge_small_adj_close_groups(5, 500))
    self.assertEqual(group.num_groups(), 1)

    # Another round of merge call can't find another merge candidate, returns
    # true indicating no more iterations are needed.
    self.assertTrue(group.merge_small_adj_close_groups(5, 500))

  def test_breakup_groups(self):
    meta_netlist = copy.deepcopy(self._meta_netlist)
    group = grouping.Grouping(meta_netlist)
    name_to_id_map = {node.name: node.id for node in meta_netlist.node}
    s0_id = name_to_id_map["S0"]
    s1_id = name_to_id_map["S1"]
    group.set_node_group(s0_id, 3)
    group.set_node_group(s1_id, 3)

    meta_netlist.node[s0_id].coord = mnds.Coord(x=0, y=0)
    meta_netlist.node[s1_id].coord = mnds.Coord(x=0, y=100)

    group.breakup_groups(100)
    self.assertEqual(group.num_groups(), 1)
    group.breakup_groups(90)
    self.assertEqual(group.num_groups(), 2)
    self.assertEqual(group.group_ids()[0], 4)
    self.assertEqual(group.group_ids()[1], 5)

    self.assertFalse(group.merge_small_adj_close_groups(5, 500))
    self.assertEqual(group.num_groups(), 1)


if __name__ == "__main__":
  absltest.main()
