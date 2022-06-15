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
"""Tests for placement_util_non_prod."""

import os

from absl import flags
from circuit_training.environment import placement_util
from circuit_training.utils import test_utils

# Internal gfile dependencies

FLAGS = flags.FLAGS

TEST_FILE_BODY = r"""# Placement file for Circuit Training
# Source input file(s) : /input/netlist.pb.txt
# This file : /this/file/initial.plc
# Original initial placement : /original/initial.plc
# Columns : 30  Rows : 20
# Width : 3000.0  Height : 2000.0
# Project : viperfish
# Block : fp_test_1
# Blockage : 0.0 100.0 3000.0 300.0 1.0
# Blockage : 3000.0 0.0 500.0 2000.0 1.0

"""


class MockPlacementCost(object):
  """A Mock class of PlacementCost for testing."""

  def __init__(self):
    self.node_type = [
        'PORT', 'PORT', 'MACRO_PIN', 'MACRO_PIN', 'MACRO', 'STDCELL'
    ]
    self._fix_node_coord = [False] * len(self.node_type)

  def get_node_type(self, node: int):
    if node >= len(self.node_type):
      return None
    return self.node_type[node]

  def fix_node_coord(self, node: int):
    self._fix_node_coord[node] = True

  def get_grid_num_columns_rows(self):
    return (10, 12)

  def get_canvas_width_height(self):
    return (100.0, 120.0)

  def get_routes_per_micron(self):
    return (1.0, 2.0)

  def get_macro_routing_allocation(self):
    return (3.0, 4.0)

  def get_congestion_smooth_range(self):
    return 2.0

  def get_source_filename(self):
    return '/source/filename'

  def get_area(self):
    return 10

  def get_wirelength(self):
    return 11.0

  def get_cost(self):
    return 12.0

  def get_congestion_cost(self):
    return 13.0

  def get_density_cost(self):
    return 14.0

  def get_project_name(self):
    return 'project'

  def get_block_name(self):
    return 'block'

  def get_overlap_threshold(self):
    return 1e-6

  def get_blockages(self):
    return [[0, 0, 10.0, 10.0], [0, 20.0, 10.0, 30.0]]

  def get_ref_node_id(self, node_id):
    del node_id
    return -1

  def is_node_soft_macro(self, node_id):
    del node_id
    return False

  def get_node_location(self, node_id):
    if node_id == 0:
      return (0, 0)
    elif node_id == 1:
      return (0, 1)
    elif node_id == 2:
      return (0, 2)
    elif node_id == 3:
      return (0, 3)
    elif node_id == 4:
      return (0, 4)
    elif node_id == 5:
      return (0, 5)

  def get_macro_orientation(self, node_id):
    if node_id == 4:
      return 'N'
    return ''

  def is_node_placed(self, node_id):
    del node_id
    return True

  def save_placement(self, filename, info):
    print(info)
    with open(filename, 'wt') as f:
      for l in info.split('\n'):
        f.write('# ' + l + '\n')

  def get_grid_cell_of_node(self, index):
    return 0

  def unplace_all_nodes(self):
    return

  def is_node_fixed(self, index):
    return True


class PlacementUtilTest(test_utils.TestCase):

  def setUp(self):
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/macro_tiles_10x10')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    self.plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=init_placement)

  def test_mock_plc_get_node_type(self):
    plc = MockPlacementCost()
    self.assertEqual(list(placement_util.nodes_of_types(plc, ['PORT'])), [0, 1])
    self.assertEqual(
        list(placement_util.nodes_of_types(plc, ['MACRO_PIN'])), [2, 3])
    self.assertEqual(list(placement_util.nodes_of_types(plc, ['MACRO'])), [4])
    self.assertEqual(
        list(placement_util.nodes_of_types(plc, ['PORT', 'MACRO'])), [0, 1, 4])
    self.assertEmpty(list(placement_util.nodes_of_types(plc, ['BAD_TYPE'])))

  def test_mock_plc_get_node_xy_coordinates(self):
    plc = MockPlacementCost()
    # This function returns only PORT, MACRO, and STDCELL nodes.
    self.assertDictEqual(
        placement_util.get_node_xy_coordinates(plc), {
            0: (0, 0),
            1: (0, 1),
            4: (0, 4),
            5: (0, 5)
        })

  def test_mock_plc_get_macro_orientations(self):
    plc = MockPlacementCost()
    # This function returns only MACRO.
    self.assertDictEqual(placement_util.get_macro_orientations(plc), {4: 'N'})

  def test_mock_plc_fix_port_coordinates(self):
    plc = MockPlacementCost()
    placement_util.fix_port_coordinates(plc)
    self.assertTrue(plc._fix_node_coord[0])
    self.assertTrue(plc._fix_node_coord[1])
    self.assertFalse(plc._fix_node_coord[2])
    self.assertFalse(plc._fix_node_coord[3])

  def test_sample_file_extract_attribute(self):
    tempfile = self.create_tempfile().full_path
    with open(tempfile, 'wt') as f:
      f.write(TEST_FILE_BODY)
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Block', [tempfile]),
        'fp_test_1')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Project', [tempfile]),
        'viperfish')
    self.assertIsNone(
        placement_util.extract_attribute_from_comments('Unknown_Atrribute',
                                                       [tempfile]))

  def test_sample_file_extract_parameters(self):
    tempfile = self.create_tempfile().full_path
    with open(tempfile, 'wt') as f:
      f.write(TEST_FILE_BODY)

    sizes = placement_util.extract_sizes_from_comments([tempfile])
    self.assertLen(sizes, 4)
    canvas_width, canvas_height, grid_cols, grid_rows = sizes
    self.assertEqual(canvas_width, 3000.0)
    self.assertEqual(canvas_height, 2000.0)
    self.assertEqual(grid_cols, 30)
    self.assertEqual(grid_rows, 20)

  def test_sample_file_get_blockages(self):
    tempfile = self.create_tempfile().full_path
    with open(tempfile, 'wt') as f:
      f.write(TEST_FILE_BODY)
    blockages = placement_util.get_blockages_from_comments([tempfile])

    self.assertLen(blockages, 2)
    self.assertEqual(blockages[0], [0.0, 100.0, 3000.0, 300.0, 1.0])
    self.assertEqual(blockages[1], [3000.0, 0.0, 500.0, 2000.0, 1.0])

  def test_save_placement(self):
    filename = os.path.join(self.create_tempdir(), 'placement.plc')
    plc = MockPlacementCost()
    placement_util.save_placement(plc, filename, 'user_comments')

    sizes = placement_util.extract_sizes_from_comments([filename])
    self.assertEqual(sizes, (100.0, 120.0, 10, 12))
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Area', [filename]),
        '10')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Wirelength',
                                                       [filename]), '11')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Wirelength cost',
                                                       [filename]), '12')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Congestion cost',
                                                       [filename]), '13')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Density cost',
                                                       [filename]), '14')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Project', [filename]),
        'project')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Block', [filename]),
        'block')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Smoothing factor',
                                                       [filename]), '2')
    self.assertEqual(
        placement_util.extract_attribute_from_comments('Overlap threshold',
                                                       [filename]), '1e-06')

    self.assertEqual(
        placement_util.get_blockages_from_comments([filename]),
        [[0, 0, 10.0, 10.0], [0, 20.0, 10.0, 30.0]])

  def test_sample_netlist_create_plc(self):
    """Test creating placement cost with sample netlist.

    # Internal circuit training docs link.
    """

    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/macro_tiles_10x10')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=init_placement)

    self.assertEqual(plc.get_canvas_width_height(), (1200, 1200))
    self.assertEqual(plc.get_grid_num_columns_rows(), (20, 20))
    self.assertEqual(plc.get_project_name(), 'circuit_training')
    self.assertEqual(plc.get_block_name(), 'macro_tiles_10x10')
    self.assertEqual(plc.get_routes_per_micron(), (70.33, 74.51))
    self.assertEqual(plc.get_macro_routing_allocation(), (51.79, 51.79))
    self.assertEqual(plc.get_congestion_smooth_range(), 2.0)
    self.assertEqual(plc.get_overlap_threshold(), 4e-3)
    self.assertFalse(plc.get_canvas_boundary_check())

    self.assertGreater(plc.get_cost(), 0.0)

  def test_sample_netlist_run_fd(self):
    """Test running FD on a sample netlist.

    # Internal circuit training docs link.
    """
    self.assertGreater(self.plc.get_cost(), 0.0)
    placement_util.fd_placement_schedule(self.plc)
    self.assertGreater(self.plc.get_cost(), 0.0)

  def test_legalize_placement(self):
    self.assertTrue(placement_util.legalize_placement(self.plc))

  def test_disconnect_high_fanout_nets(self):
    placement_util.disconnect_high_fanout_nets(self.plc, 500)

  def test_create_placement_cost_using_common_arguments(self):
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/macro_tiles_10x10')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    plc = placement_util.create_placement_cost_using_common_arguments(
        netlist_file, init_placement, 100, 100, 10, 10)
    self.assertEqual(plc.get_canvas_width_height(), (100, 100))
    self.assertEqual(plc.get_grid_num_columns_rows(), (10, 10))

  def test_save_placement_with_info(self):
    output_file_path = os.path.join(FLAGS.test_tmpdir, 'netlist.pb.txt')
    self.assertEqual(
        placement_util.save_placement_with_info(self.plc, output_file_path,
                                                'python run'), {
                                                    'message': '',
                                                    'ok': True
                                                })

  def test_get_hard_macro_density_map(self):
    self.assertEqual(
        placement_util.get_hard_macro_density_map(self.plc), [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0.6944444444444444, 0.6944444444444444, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6944444444444444, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0.6944444444444444, 0,
            0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.6944444444444444, 0.6944444444444444, 0, 0.6944444444444444,
            0.6944444444444444, 0.6944444444444444, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ])

  def test_extract_parameters_from_comments(self):
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/macro_tiles_10x10')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    self.assertEqual(
        placement_util.extract_parameters_from_comments(init_placement),
        (1200.0, 1200.0, 20, 20))


if __name__ == '__main__':
  test_utils.main()
