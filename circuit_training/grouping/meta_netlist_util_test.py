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
"""Tests for meta_netlist_util."""

from absl.testing import absltest
from circuit_training.grouping import meta_netlist_convertor
from circuit_training.grouping import meta_netlist_util
from circuit_training.utils import test_utils

_NETLIST_FILE_PATH = 'third_party/py/circuit_training/grouping/testdata/simple.pb.txt'


class MetaNetlistUtilTest(absltest.TestCase):

  def test_set_canvas_width_height(self):
    meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)
    meta_netlist_util.set_canvas_width_height(meta_netlist, 11, 10)
    self.assertAlmostEqual(meta_netlist.canvas.dimension.width, 11)
    self.assertAlmostEqual(meta_netlist.canvas.dimension.height, 10)

  def test_set_canvas_cols_rows(self):
    meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)
    meta_netlist_util.set_canvas_columns_rows(meta_netlist, 11, 10)
    self.assertAlmostEqual(meta_netlist.canvas.num_columns, 11)
    self.assertAlmostEqual(meta_netlist.canvas.num_rows, 10)

  def test_disconnect_single_net(self):
    meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)
    meta_netlist_util.disconnect_single_net(meta_netlist, 0)
    self.assertEmpty(meta_netlist.node[2].input_indices)

  def test_disconnect_high_fanout_nets(self):
    meta_netlist = meta_netlist_convertor.read_netlist(_NETLIST_FILE_PATH)
    meta_netlist_util.disconnect_high_fanout_nets(meta_netlist, 1)

    # Max length for the output_indices is 1.
    for node in meta_netlist.node:
      self.assertLessEqual(len(node.output_indices), 1)


if __name__ == '__main__':
  test_utils.main()
