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
"""Testing plc converter."""

import os

from absl import logging
from absl.testing import absltest
from circuit_training.dreamplace import plc_converter
from circuit_training.environment import placement_util
from circuit_training.utils import test_utils
from dreamplace import Params
import numpy as np

_CIRCUIT_TRAINING_DIR = 'circuit_training'


class PlcConverterTest(test_utils.TestCase):

  def get_params_for_test(self, plc):
    params = Params.Params()
    params.num_bins_x, params.num_bins_y = plc.get_grid_num_columns_rows()
    params.legalize_flag = False
    params.random_center_init_flag = False
    params.enable_fillers = False
    params.circuit_training_mode = True
    params.scale_factor = 1.0
    return params

  def assert_array_of_array_equal(self, x, y):
    self.assertEqual(len(x), len(y))
    for a, b in zip(x, y):
      np.testing.assert_array_almost_equal(a, b)

  def test_convert(self):
    test_netlist_dir = os.path.join(
        _CIRCUIT_TRAINING_DIR,
        'environment/test_data/simple_grouped_with_coords',
    )
    netlist_file = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'netlist.pb.txt'
    )
    init_placement = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'initial.plc'
    )

    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=init_placement
    )
    converter = plc_converter.PlcConverter(regioning=True)
    grp_2 = plc.get_node_index('Grp_2')
    plc.add_area_constraint([grp_2], [0, 0, 400, 400])
    plc.add_area_constraint([grp_2], [400, 400, 500, 500])
    placedb = converter.convert(plc)
    placedb(self.get_params_for_test(plc))

    self.assertEqual(placedb.dtype, np.float32)
    self.assertEqual(placedb.num_physical_nodes, 6)
    self.assertEqual(placedb.num_terminals, 5)
    self.assertEqual(placedb.num_non_movable_macros, 2)
    self.assertEqual(
        placedb.node_name2id_map,
        {
            'Grp_2': 0,
            'M0': 1,
            'M1': 2,
            'P0': 3,
            'P1': 4,
            'P2': 5,
        },
    )
    np.testing.assert_array_equal(
        placedb.node_names, [b'Grp_2', b'M0', b'M1', b'P0', b'P1', b'P2']
    )
    np.testing.assert_array_almost_equal(
        placedb.node_x + placedb.node_size_x / 2, [125, 125, 375, 0, 499, 0]
    )
    np.testing.assert_array_almost_equal(
        placedb.node_y + placedb.node_size_y / 2, [125, 125, 375, 100, 499, 101]
    )
    np.testing.assert_array_equal(
        placedb.node_orient, [b'N', b'N', b'N', b'N', b'N', b'N']
    )
    np.testing.assert_array_almost_equal(
        placedb.node_size_x, [1.8795, 120, 80, 0, 0, 0]
    )
    np.testing.assert_array_almost_equal(
        placedb.node_size_y, [1.8795, 120, 40, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        placedb.pin_direct,
        [
            b'OUTPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'INPUT',
        ],
    )
    np.testing.assert_array_almost_equal(
        placedb.pin_offset_x,
        [
            0.93975,
            0.0,
            120.0,
            0.93975,
            80.0,
            0.0,
            0.0,
            0.93975,
            0.0,
            0.0,
            0.93975,
            0.0,
        ],
    )
    np.testing.assert_array_almost_equal(
        placedb.pin_offset_y,
        [
            0.93975,
            40,
            120,
            0.93975,
            40,
            0,
            0,
            0.93975,
            120,
            0,
            0.93975,
            120,
        ],
    )
    self.assertEqual(
        placedb.net_name2id_map,
        {
            'Grp_2/Poutput_single_0': 0,
            'P1_M0': 1,
            'P1_M1': 2,
            'P0': 3,
            'P2': 4,
        },
    )
    np.testing.assert_array_equal(
        placedb.net_names,
        [b'Grp_2/Poutput_single_0', b'P1_M0', b'P1_M1', b'P0', b'P2'],
    )
    self.assert_array_of_array_equal(
        placedb.net2pin_map, [[0, 1], [2, 3], [4, 5], [6, 7, 8], [9, 10, 11]]
    )
    np.testing.assert_array_equal(
        placedb.flat_net2pin_map, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    np.testing.assert_array_equal(
        placedb.flat_net2pin_start_map, [0, 2, 4, 6, 9, 12]
    )
    np.testing.assert_array_equal(
        placedb.net_weights, [1.0, 1.0, 1.0, 1.0, 1.0]
    )
    self.assert_array_of_array_equal(
        placedb.node2pin_map, [[0, 3, 7, 10], [2, 8, 11], [1, 4], [6], [5], [9]]
    )
    np.testing.assert_array_equal(
        placedb.flat_node2pin_map, [0, 3, 7, 10, 2, 8, 11, 1, 4, 6, 5, 9]
    )
    np.testing.assert_array_equal(
        placedb.flat_node2pin_start_map, [0, 4, 7, 9, 10, 11, 12]
    )
    np.testing.assert_array_equal(
        placedb.pin2node_map, [0, 2, 1, 0, 2, 4, 3, 0, 1, 5, 0, 1]
    )
    np.testing.assert_array_equal(
        placedb.pin2net_map, [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]
    )
    np.testing.assert_array_almost_equal(
        placedb.rows, [[0.0, 0.0, 500.0, 250.0], [0.0, 250.0, 500.0, 500.0]]
    )
    self.assertEqual(placedb.xl, 0)
    self.assertEqual(placedb.yl, 0)
    self.assertEqual(placedb.xh, 500)
    self.assertEqual(placedb.yh, 500)
    self.assertEqual(placedb.row_height, 250)
    self.assertEqual(placedb.site_width, 250)
    self.assertEqual(placedb.num_movable_pins, 4)
    self.assertEqual(placedb.bin_size_x, 250)
    self.assertEqual(placedb.bin_size_y, 250)
    self.assertEqual(placedb.num_bins_x, 2)
    self.assertEqual(placedb.num_bins_y, 2)
    logging.info('Pass first part tests.')
    np.testing.assert_array_almost_equal(placedb.bin_center_x, [125.0, 375.0])
    np.testing.assert_array_almost_equal(placedb.bin_center_y, [125.0, 375.0])
    self.assertAlmostEqual(
        placedb.total_movable_node_area, 3.53252029, places=4
    )
    self.assertAlmostEqual(placedb.total_fixed_node_area, 17600.0)
    self.assertAlmostEqual(placedb.total_filler_node_area, 0)
    self.assertAlmostEqual(placedb.total_space_area, 232400.0)
    self.assertEqual(placedb.num_filler_nodes, 0)
    np.testing.assert_array_equal(placedb.macro_mask, [0])

    self.assertEqual(
        converter.node_index_to_node_id_map,
        {0: 3, 1: 4, 2: 5, 3: 1, 4: 2, 9: 0},
    )
    self.assertEqual(converter.driver_pin_indices, [10, 6, 8, 0, 2])
    self.assertEqual(
        converter.pin_id_to_pin_index, [10, 7, 6, 11, 8, 1, 0, 11, 5, 2, 11, 5]
    )
    self.assertEqual(converter.soft_macro_and_stdcell_indices, [9])
    self.assertEqual(converter.hard_macro_indices, [3, 4])
    self.assertEqual(converter.non_movable_node_indices, [0, 1, 2])

    # Test update_macro by updating a hard macro in plc.
    macro_index = plc.get_node_index('M0')
    plc.update_node_coords(macro_index, 200, 200)
    plc.update_macro_orientation(macro_index, 'S')
    converter.update_macro(placedb, plc, macro_index)

    np.testing.assert_array_almost_equal(
        placedb.node_x + placedb.node_size_x / 2,
        [125.0, 200.0, 375.0, 0.0, 499.0, 0.0],
    )
    np.testing.assert_array_almost_equal(
        placedb.node_y + placedb.node_size_y / 2, [125, 200, 375, 100, 499, 101]
    )
    np.testing.assert_array_equal(
        placedb.node_orient, [b'N', b'S', b'N', b'N', b'N', b'N']
    )
    np.testing.assert_array_almost_equal(
        placedb.pin_offset_x,
        [
            0.93975,
            0.0,
            0.0,
            0.93975,
            80.0,
            0.0,
            0.0,
            0.93975,
            120.0,
            0.0,
            0.93975,
            120.0,
        ],
    )
    np.testing.assert_array_almost_equal(
        placedb.pin_offset_y,
        [
            0.93975,
            40.0,
            0.0,
            0.93975,
            40.0,
            0.0,
            0.0,
            0.93975,
            0.0,
            0.0,
            0.93975,
            0.0,
        ],
    )

    self.assert_array_of_array_equal(
        placedb.regions, np.array([[[0, 0, 400, 400], [400, 400, 500, 500]]])
    )
    np.testing.assert_array_equal(
        placedb.flat_region_boxes,
        np.array([0, 0, 400, 400, 400, 400, 500, 500], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        placedb.flat_region_boxes_start, np.array([0, 8], dtype=np.int32)
    )
    print('placedb.node2fence_region_map: ', placedb.node2fence_region_map)
    np.testing.assert_array_equal(
        placedb.node2fence_region_map,
        np.array([0], dtype=np.int32),
    )

  def test_mixed_size_with_changing_hard_macro(self):
    test_netlist_dir = os.path.join(
        _CIRCUIT_TRAINING_DIR,
        'environment/test_data/simple_grouped_with_coords',
    )
    netlist_file = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'netlist.pb.txt'
    )
    init_placement = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'initial.plc'
    )

    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=init_placement
    )
    converter = plc_converter.PlcConverter()
    placedb = converter.convert(plc)
    placedb(self.get_params_for_test(plc))

    # Let's fix one macro and allow DP to move the other.
    converter.update_num_non_movable_macros(
        placedb, plc, num_non_movable_macros=1
    )
    placedb(self.get_params_for_test(plc))

    self.assertEqual(placedb.num_terminals, 4)
    np.testing.assert_array_equal(placedb.macro_mask, [0, 1])
    self.assertAlmostEqual(placedb.total_space_area, 235600.0)
    np.testing.assert_array_almost_equal(
        placedb.node_x + placedb.node_size_x / 2, [125, 125, 375, 0, 499, 0]
    )
    np.testing.assert_array_almost_equal(
        placedb.node_y + placedb.node_size_y / 2, [125, 125, 375, 100, 499, 101]
    )
    self.assertEqual(placedb.num_movable_pins, 7)
    self.assertEqual(placedb.num_non_movable_macros, 1)

  def test_change_macro_order(self):
    test_netlist_dir = os.path.join(
        _CIRCUIT_TRAINING_DIR,
        'environment/test_data/simple_grouped_with_coords',
    )
    netlist_file = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'netlist.pb.txt'
    )
    init_placement = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'initial.plc'
    )

    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=init_placement
    )
    converter = plc_converter.PlcConverter()
    hard_macro_order = [plc.get_node_index('M1'), plc.get_node_index('M0')]
    placedb = converter.convert(plc, hard_macro_order)
    placedb(self.get_params_for_test(plc))

    # Let's fix one macro and allow DP to move the other.
    converter.update_num_non_movable_macros(
        placedb, plc, num_non_movable_macros=1
    )
    placedb(self.get_params_for_test(plc))

    self.assertEqual(placedb.num_terminals, 4)
    self.assertEqual(
        placedb.node_name2id_map,
        {
            'Grp_2': 0,
            'M1': 1,
            'M0': 2,
            'P0': 3,
            'P1': 4,
            'P2': 5,
        },
    )
    np.testing.assert_array_equal(
        placedb.node_names, [b'Grp_2', b'M1', b'M0', b'P0', b'P1', b'P2']
    )
    np.testing.assert_array_almost_equal(
        placedb.node_x + placedb.node_size_x / 2, [125, 375, 125, 0, 499, 0]
    )
    np.testing.assert_array_almost_equal(
        placedb.node_y + placedb.node_size_y / 2, [125, 375, 125, 100, 499, 101]
    )

  def test_blockage(self):
    """It is the same as test_convert except that there are two blockages in the circuit."""
    test_netlist_dir = os.path.join(
        _CIRCUIT_TRAINING_DIR,
        'environment/test_data/simple_grouped_with_coords_with_blockage',
    )
    netlist_file = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'netlist.pb.txt'
    )
    init_placement = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'initial.plc'
    )

    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=init_placement
    )
    converter = plc_converter.PlcConverter()
    placedb = converter.convert(plc)
    placedb(self.get_params_for_test(plc))

    self.assertEqual(placedb.dtype, np.float32)
    self.assertEqual(placedb.num_physical_nodes, 8)
    self.assertEqual(placedb.num_terminals, 7)
    self.assertEqual(
        placedb.node_name2id_map,
        {
            'Grp_2': 0,
            'M0': 1,
            'M1': 2,
            'P0': 3,
            'P1': 4,
            'P2': 5,
            'blockage_dummy_node_0': 6,
            'blockage_dummy_node_1': 7,
        },
    )
    np.testing.assert_array_equal(
        placedb.node_names,
        [
            b'Grp_2',
            b'M0',
            b'M1',
            b'P0',
            b'P1',
            b'P2',
            b'blockage_dummy_node_0',
            b'blockage_dummy_node_1',
        ],
    )
    np.testing.assert_array_almost_equal(
        placedb.node_x + placedb.node_size_x / 2,
        [125, 125, 375, 0, 499, 0, 5, 495],
    )
    np.testing.assert_array_almost_equal(
        placedb.node_y + placedb.node_size_y / 2,
        [125, 125, 375, 100, 499, 101, 5, 495],
    )
    np.testing.assert_array_equal(
        placedb.node_orient, [b'N', b'N', b'N', b'N', b'N', b'N', b'N', b'N']
    )
    np.testing.assert_array_almost_equal(
        placedb.node_size_x, [1.879, 120.0, 80.0, 0.0, 0.0, 0.0, 10.0, 10.0]
    )
    np.testing.assert_array_almost_equal(
        placedb.node_size_y, [1.879, 120.0, 40.0, 0.0, 0.0, 0.0, 10.0, 10.0]
    )
    self.assertAlmostEqual(placedb.total_movable_node_area, 3.530640, places=4)
    self.assertAlmostEqual(placedb.total_fixed_node_area, 17800.0)
    self.assertAlmostEqual(placedb.total_space_area, 232200.0)

  def test_convert_stdcells(self):
    test_netlist_dir = os.path.join(
        _CIRCUIT_TRAINING_DIR,
        'environment/test_data/simple_with_coords',
    )
    netlist_file = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'netlist.pb.txt'
    )
    init_placement = os.path.join(
        absltest.TEST_SRCDIR.value, test_netlist_dir, 'initial.plc'
    )

    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=init_placement
    )
    plc.set_placement_grid(2, 2)
    converter = plc_converter.PlcConverter()
    placedb = converter.convert(plc)
    placedb(self.get_params_for_test(plc))

    self.assertEqual(placedb.dtype, np.float32)
    self.assertEqual(placedb.num_physical_nodes, 6)
    self.assertEqual(placedb.num_terminals, 4)
    self.assertEqual(
        placedb.node_name2id_map,
        {'S0': 0, 'S1': 1, 'M0': 2, 'M1': 3, 'P0': 4, 'P1': 5},
    )
    np.testing.assert_array_equal(
        placedb.node_names, [b'S0', b'S1', b'M0', b'M1', b'P0', b'P1']
    )
    np.testing.assert_array_almost_equal(
        placedb.node_x + placedb.node_size_x / 2, [100, 150, 100, 300, 0, 500]
    )
    np.testing.assert_array_almost_equal(
        placedb.node_y + placedb.node_size_y / 2, [400, 400, 100, 100, 100, 500]
    )
    np.testing.assert_array_equal(
        placedb.node_orient, [b'N', b'N', b'N', b'N', b'N', b'N']
    )
    np.testing.assert_array_almost_equal(
        placedb.node_size_x, [2.208, 2.208, 120, 80, 0, 0]
    )
    np.testing.assert_array_almost_equal(
        placedb.node_size_y, [0.48, 0.48, 120, 40, 0, 0]
    )
    np.testing.assert_array_equal(
        placedb.pin_direct,
        [
            b'OUTPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'OUTPUT',
            b'INPUT',
            b'INPUT',
        ],
    )
    np.testing.assert_array_almost_equal(
        placedb.pin_offset_x,
        [1.104, 1.104, 1.104, 0.0, 120.0, 1.104, 80.0, 0.0, 0.0, 0.0, 1.104],
    )
    np.testing.assert_array_almost_equal(
        placedb.pin_offset_y,
        [0.24, 0.24, 0.24, 40.0, 120.0, 0.24, 40.0, 0.0, 0.0, 120.0, 0.24],
    )
    self.assertAlmostEqual(placedb.total_movable_node_area, 2.1196799278259277)
    self.assertAlmostEqual(placedb.total_fixed_node_area, 17600.0)
    self.assertAlmostEqual(placedb.total_space_area, 232400.0)
    np.testing.assert_array_equal(placedb.macro_mask, [0, 0])


if __name__ == '__main__':
  test_utils.main()
