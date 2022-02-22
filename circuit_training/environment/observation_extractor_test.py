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
"""Tests for circuit_training.environment.observation_extractor."""

import os

from absl import flags
from absl import logging
from circuit_training.environment import observation_config
from circuit_training.environment import observation_extractor
from circuit_training.environment import placement_util
from circuit_training.utils import test_utils
import numpy as np

FLAGS = flags.FLAGS


class ObservationExtractorTest(test_utils.TestCase):
  """Tests for the ObservationExtractor.

  # Internal circuit training docs link.
  """

  def setUp(self):
    super(ObservationExtractorTest, self).setUp()

    self._observation_config = observation_config.ObservationConfig(
        max_num_edges=8, max_num_nodes=6, max_grid_size=10)

    # Macros name                      : M0, M1, Grp_2
    # Order in plc.get_macro_indices():  0,  1,  2
    # Edges: (0, 1), (0, 2)
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/sample_clustered')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement='')
    plc.set_canvas_size(300, 200)
    plc.set_placement_grid(9, 4)
    plc.unplace_all_nodes()
    # Manually adds I/O port locations, this step is not needed for real
    # netlists.
    plc.update_node_coords('P0', 0.5, 100)  # Left
    plc.update_node_coords('P1', 150, 199.5)  # Top
    plc.update_port_sides()
    plc.snap_ports_to_edges()
    self.extractor = observation_extractor.ObservationExtractor(
        plc=plc, observation_config=self._observation_config)

  def test_static_features(self):
    static_obs = self.extractor.get_static_features()
    logging.info('static observation: %s', static_obs)
    self.assertEqual(static_obs['normalized_num_edges'], 5.0 / 8.0)
    self.assertEqual(static_obs['normalized_num_hard_macros'], 2.0 / 6.0)
    self.assertEqual(static_obs['normalized_num_soft_macros'], 1.0 / 6.0)
    self.assertEqual(static_obs['normalized_num_port_clusters'], 2.0 / 6.0)
    self.assertAllClose(static_obs['macros_w'],
                        np.asarray([120., 80., 0., 0., 0., 0.]) / 300.0)
    self.assertAllClose(static_obs['macros_h'],
                        np.asarray([120., 40., 0., 0., 0., 0.]) / 200.0)
    self.assertAllEqual(static_obs['node_types'], [1, 1, 2, 3, 3, 0])
    self.assertAllEqual(static_obs['sparse_adj_i'], [0, 0, 1, 1, 2, 0, 0, 0])
    self.assertAllEqual(static_obs['sparse_adj_j'], [2, 3, 2, 4, 3, 0, 0, 0])
    # Graph Description:
    # 0->2, 0->3, 1->2, 1->4, 2->3
    # Node 0: two edges
    # Node 1: two edges
    # Node 2: three edges
    # Node 3: two edges
    # Node 4: one edge
    # The last zero in the array is due to the value of `max_num_nodes`.
    self.assertAllEqual(static_obs['edge_counts'], [2, 2, 3, 2, 1, 0])
    self.assertAllClose(static_obs['sparse_adj_weight'],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0])

  def test_initial_dynamic_features(self):
    mask = np.zeros(
        self._observation_config.max_grid_size *
        self._observation_config.max_grid_size,
        dtype=np.float32)
    dynamic_obs = self.extractor.get_dynamic_features(
        previous_node_index=-1, current_node_index=0, mask=mask)
    logging.info('dynamic observation: %s', dynamic_obs)
    # Replaces the unplaced node in the middle of the canvas.
    self.assertAllClose(dynamic_obs['locations_x'],
                        np.asarray([150., 150., 150., 0., 150., 0.0]) / 300.0)
    self.assertAllClose(dynamic_obs['locations_y'],
                        np.asarray([100., 100., 100., 125., 200., 0.0]) / 200.0)
    self.assertAllEqual(dynamic_obs['is_node_placed'], [0, 0, 0, 1, 1, 0])
    self.assertAllClose(dynamic_obs['mask'],
                        [0] * (self._observation_config.max_grid_size *
                               self._observation_config.max_grid_size))
    self.assertAllClose(dynamic_obs['current_node'], [0])

  def test_initial_all_features(self):
    mask = np.zeros(
        self._observation_config.max_grid_size *
        self._observation_config.max_grid_size,
        dtype=np.float32)
    all_obs = self.extractor.get_all_features(
        previous_node_index=-1, current_node_index=0, mask=mask)
    logging.info('All observation: %s', all_obs)
    self.assertEqual(all_obs['normalized_num_edges'], 5.0 / 8.0)
    self.assertEqual(all_obs['normalized_num_hard_macros'], 2.0 / 6.0)
    self.assertEqual(all_obs['normalized_num_soft_macros'], 1.0 / 6.0)
    self.assertEqual(all_obs['normalized_num_port_clusters'], 2.0 / 6.0)
    self.assertAllClose(all_obs['macros_w'],
                        np.asarray([120., 80., 0., 0., 0., 0.]) / 300.0)
    self.assertAllClose(all_obs['macros_h'],
                        np.asarray([120., 40., 0., 0., 0., 0.]) / 200.0)
    self.assertAllEqual(all_obs['node_types'], [1, 1, 2, 3, 3, 0])
    self.assertAllEqual(all_obs['sparse_adj_i'], [0, 0, 1, 1, 2, 0, 0, 0])
    self.assertAllEqual(all_obs['sparse_adj_j'], [2, 3, 2, 4, 3, 0, 0, 0])
    self.assertAllClose(all_obs['sparse_adj_weight'],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0])
    self.assertAllClose(all_obs['locations_x'],
                        np.asarray([150., 150., 150., 0., 150., 0.0]) / 300.0)
    self.assertAllClose(all_obs['locations_y'],
                        np.asarray([100., 100., 100., 125., 200., 0.0]) / 200.0)
    self.assertAllEqual(all_obs['is_node_placed'], [0, 0, 0, 1, 1, 0])
    self.assertAllClose(all_obs['mask'],
                        [0] * (self._observation_config.max_grid_size *
                               self._observation_config.max_grid_size))
    self.assertAllClose(all_obs['current_node'], [0])

    obs_space = self._observation_config.observation_space
    self.assertTrue(obs_space.contains(all_obs))

  def test_all_features_after_step(self):
    self.extractor.plc.update_node_coords('M0', 100, 120)
    mask = np.zeros(
        self._observation_config.max_grid_size *
        self._observation_config.max_grid_size,
        dtype=np.float32)
    all_obs = self.extractor.get_all_features(
        previous_node_index=0, current_node_index=1, mask=mask)
    self.assertAllClose(all_obs['locations_x'],
                        np.asarray([100., 150., 150., 0., 150., 0.0]) / 300.0)
    self.assertAllClose(all_obs['locations_y'],
                        np.asarray([120., 100., 100., 125., 200., 0.0]) / 200.0)
    self.assertAllEqual(all_obs['is_node_placed'], [1, 0, 0, 1, 1, 0])
    self.assertAllClose(all_obs['current_node'], [1])

    self.extractor.plc.update_node_coords('M1', 200, 150)
    all_obs = self.extractor.get_all_features(
        previous_node_index=1, current_node_index=2, mask=mask)
    self.assertAllClose(all_obs['locations_x'],
                        np.asarray([100., 200., 150., 0., 150., 0.0]) / 300.0)
    self.assertAllClose(all_obs['locations_y'],
                        np.asarray([120., 150., 100., 125., 200., 0.0]) / 200.0)
    self.assertAllEqual(all_obs['is_node_placed'], [1, 1, 0, 1, 1, 0])
    self.assertAllClose(all_obs['current_node'], [2])


if __name__ == '__main__':
  test_utils.main()
