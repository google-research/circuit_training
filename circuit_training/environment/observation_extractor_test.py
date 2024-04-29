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


_CIRCUIT_TRAINING_DIR = 'circuit_training'
_TESTDATA_DIR = (
    _CIRCUIT_TRAINING_DIR + '/environment/test_data/sample_clustered/'
)


class ObservationExtractorTest(test_utils.TestCase):
  """Tests for the ObservationExtractor.

  # Internal circuit training docs link.
  """

  def setUp(self):
    super(ObservationExtractorTest, self).setUp()

    self._observation_config = observation_config.ObservationConfig(
        max_num_edges=8, max_num_nodes=6, max_grid_size=3
    )

    # Macros name                      : M0, M1, Grp_2
    # Order in plc.get_macro_indices():  0,  1,  2
    # Edges: (0, 1), (0, 2)
    netlist_file = os.path.join(
        FLAGS.test_srcdir, _TESTDATA_DIR, 'netlist.pb.txt'
    )
    plc = placement_util.create_placement_cost(
        netlist_file=netlist_file, init_placement=''
    )
    plc.set_canvas_size(300, 200)
    plc.set_placement_grid(3, 2)
    plc.unplace_all_nodes()
    # Manually adds I/O port locations, this step is not needed for real
    # netlists.
    plc.update_node_coords('P0', 0.5, 100)  # Left
    plc.fix_node_coord('P0')
    plc.update_node_coords('P1', 150, 199.5)  # Top
    plc.fix_node_coord('P1')
    plc.update_port_sides()
    plc.snap_ports_to_edges()
    # Manually adds fake net, this step is not needed for real
    # netlists.
    plc.add_fake_net(0.1, [1, 2])  # P1 <-> M0
    plc.add_fake_net(0.2, [2, 3])  # M0 <-> M1
    self.extractor = observation_extractor.ObservationExtractor(
        plc=plc, observation_config=self._observation_config, netlist_index=0
    )
    self.extractor_default_init = observation_extractor.ObservationExtractor(
        plc=plc,
        observation_config=self._observation_config,
        netlist_index=0,
        use_plc_init_location=False,
    )

  def test_static_features(self):
    static_obs = self.extractor.get_static_features()
    logging.info('static observation: %s', static_obs)
    self.assertEqual(static_obs['normalized_num_edges'], 5.0 / 8.0)
    self.assertEqual(static_obs['normalized_num_hard_macros'], 2.0 / 6.0)
    self.assertEqual(static_obs['normalized_num_soft_macros'], 1.0 / 6.0)
    self.assertEqual(static_obs['normalized_num_port_clusters'], 2.0 / 6.0)
    self.assertAllClose(
        static_obs['macros_w'],
        np.asarray([120.0, 80.0, 0.0, 0.0, 0.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        static_obs['macros_h'],
        np.asarray([120.0, 40.0, 0.0, 0.0, 0.0, 0.0]) / 500.0,
    )
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
    self.assertAllClose(
        static_obs['sparse_adj_weight'], [1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
    )

  def test_initial_dynamic_features(self):
    mask = np.zeros(
        self._observation_config.max_grid_size
        * self._observation_config.max_grid_size,
        dtype=np.float32,
    )
    dynamic_obs = self.extractor.get_dynamic_features(
        previous_node_index=-1, current_node_index=0, mask=mask
    )
    logging.info('dynamic observation: %s', dynamic_obs)
    # Replaces the unplaced node in the middle of the canvas.
    self.assertAllClose(
        dynamic_obs['locations_x'],
        np.asarray([150.0, 150.0, 150.0, 0.0, 150.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        dynamic_obs['locations_y'],
        np.asarray([100.0, 100.0, 100.0, 200.0, 200.0, 0.0]) / 500.0,
    )
    self.assertAllEqual(dynamic_obs['is_node_placed'], [0, 0, 0, 1, 1, 0])
    # The max corresponds to the closest grid to P1.
    self.assertAllClose(
        dynamic_obs['fake_net_heatmap'],
        [0.0, 0.292893, 0.0, 0.292893, 1.0, 0.292893, 0.0, 0.0, 0.0],
    )
    self.assertAllClose(
        dynamic_obs['packing_heatmap'],
        [0]
        * (
            self._observation_config.max_grid_size
            * self._observation_config.max_grid_size
        ),
    )
    self.assertAllClose(
        dynamic_obs['mask'],
        [0]
        * (
            self._observation_config.max_grid_size
            * self._observation_config.max_grid_size
        ),
    )
    self.assertAllClose(dynamic_obs['current_node'], [0])
    self.assertEqual(dynamic_obs['netlist_index'][0], 0)

  def test_initial_all_features(self):
    mask = np.zeros(
        self._observation_config.max_grid_size
        * self._observation_config.max_grid_size,
        dtype=np.float32,
    )
    all_obs = self.extractor.get_all_features(
        previous_node_index=-1, current_node_index=0, mask=mask
    )
    logging.info('All observation: %s', all_obs)
    self.assertEqual(all_obs['normalized_num_edges'], 5.0 / 8.0)
    self.assertEqual(all_obs['normalized_num_hard_macros'], 2.0 / 6.0)
    self.assertEqual(all_obs['normalized_num_soft_macros'], 1.0 / 6.0)
    self.assertEqual(all_obs['normalized_num_port_clusters'], 2.0 / 6.0)
    self.assertAllClose(
        all_obs['macros_w'],
        np.asarray([120.0, 80.0, 0.0, 0.0, 0.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        all_obs['macros_h'],
        np.asarray([120.0, 40.0, 0.0, 0.0, 0.0, 0.0]) / 500.0,
    )
    self.assertAllEqual(all_obs['node_types'], [1, 1, 2, 3, 3, 0])
    self.assertAllEqual(all_obs['sparse_adj_i'], [0, 0, 1, 1, 2, 0, 0, 0])
    self.assertAllEqual(all_obs['sparse_adj_j'], [2, 3, 2, 4, 3, 0, 0, 0])
    self.assertAllClose(
        all_obs['sparse_adj_weight'], [1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
    )
    self.assertAllClose(
        all_obs['locations_x'],
        np.asarray([150.0, 150.0, 150.0, 0.0, 150.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        all_obs['locations_y'],
        np.asarray([100.0, 100.0, 100.0, 200.0, 200.0, 0.0]) / 500.0,
    )
    self.assertAllEqual(all_obs['is_node_placed'], [0, 0, 0, 1, 1, 0])
    # The max corresponds to the closest grid to P1.
    self.assertAllClose(
        all_obs['fake_net_heatmap'],
        [0.0, 0.292893, 0.0, 0.292893, 1.0, 0.292893, 0.0, 0.0, 0.0],
    )
    self.assertAllClose(
        all_obs['packing_heatmap'],
        [0]
        * (
            self._observation_config.max_grid_size
            * self._observation_config.max_grid_size
        ),
    )
    self.assertAllClose(
        all_obs['mask'],
        [0]
        * (
            self._observation_config.max_grid_size
            * self._observation_config.max_grid_size
        ),
    )
    self.assertAllClose(all_obs['current_node'], [0])
    self.assertEqual(all_obs['netlist_index'][0], 0)

    obs_space = self._observation_config.observation_space
    self.assertTrue(obs_space.contains(all_obs))

  def test_initial_features_after_reset(self):
    mask = np.zeros(
        self._observation_config.max_grid_size
        * self._observation_config.max_grid_size,
        dtype=np.float32,
    )
    initial_obs = self.extractor.get_all_features(
        previous_node_index=-1, current_node_index=0, mask=mask
    )
    self.extractor.plc.update_node_coords('M0', 100, 120)
    _ = self.extractor.get_all_features(
        previous_node_index=0, current_node_index=1, mask=mask
    )
    self.extractor.reset()
    self.extractor.plc.unplace_all_nodes()
    initial_obs_after_reset = self.extractor.get_all_features(
        previous_node_index=-1, current_node_index=0, mask=mask
    )
    for k in initial_obs:
      self.assertAllClose(initial_obs[k], initial_obs_after_reset[k])

  def test_all_features_after_step(self):
    self.extractor.plc.update_node_coords('M0', 100, 120)
    mask = np.zeros(
        self._observation_config.max_grid_size
        * self._observation_config.max_grid_size,
        dtype=np.float32,
    )
    all_obs1 = self.extractor.get_all_features(
        previous_node_index=0, current_node_index=1, mask=mask
    )
    self.assertAllClose(
        all_obs1['locations_x'],
        np.asarray([100.0, 150.0, 150.0, 0.0, 150.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        all_obs1['locations_y'],
        np.asarray([120.0, 100.0, 100.0, 200.0, 200.0, 0.0]) / 500.0,
    )
    self.assertAllEqual(all_obs1['is_node_placed'], [1, 0, 0, 1, 1, 0])
    self.assertAllClose(all_obs1['current_node'], [1])
    # The max corresponds to the closest grid to M0.
    self.assertAllClose(
        all_obs1['fake_net_heatmap'],
        [0.0, 0.292893, 0.0, 0.292893, 1.0, 0.292893, 0.0, 0.0, 0.0],
    )
    self.assertAllClose(
        all_obs1['packing_heatmap'],
        [0]
        * (
            self._observation_config.max_grid_size
            * self._observation_config.max_grid_size
        ),
    )

    self.extractor.plc.update_node_coords('M1', 200, 150)
    all_obs2 = self.extractor.get_all_features(
        previous_node_index=1, current_node_index=2, mask=mask
    )
    self.assertAllClose(
        all_obs2['locations_x'],
        np.asarray([100.0, 200.0, 150.0, 0.0, 150.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        all_obs2['locations_y'],
        np.asarray([120.0, 150.0, 100.0, 200.0, 200.0, 0.0]) / 500.0,
    )
    self.assertAllEqual(all_obs2['is_node_placed'], [1, 1, 0, 1, 1, 0])
    self.assertAllClose(all_obs2['current_node'], [2])
    self.assertEqual(all_obs2['netlist_index'][0], 0)
    self.assertAllClose(
        all_obs2['fake_net_heatmap'],
        [0]
        * (
            self._observation_config.max_grid_size
            * self._observation_config.max_grid_size
        ),
    )

    # Also, ensure `all_obs1` is not modified.
    self.assertAllClose(
        all_obs1['locations_x'],
        np.asarray([100.0, 150.0, 150.0, 0.0, 150.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        all_obs1['locations_y'],
        np.asarray([120.0, 100.0, 100.0, 200.0, 200.0, 0.0]) / 500.0,
    )
    self.assertAllEqual(all_obs1['is_node_placed'], [1, 0, 0, 1, 1, 0])
    self.assertAllClose(all_obs1['current_node'], [1])

  def test_disable_init_locations(self):
    mask = np.zeros(
        self._observation_config.max_grid_size
        * self._observation_config.max_grid_size,
        dtype=np.float32,
    )
    all_obs = self.extractor_default_init.get_all_features(
        previous_node_index=-1, current_node_index=0, mask=mask
    )

    # Only three macros use the default locations. The rest is clustered_port.
    self.assertAllClose(
        all_obs['locations_x'],
        np.asarray([150.0, 150.0, 150.0, 0.0, 150.0, 0.0]) / 500.0,
    )
    self.assertAllClose(
        all_obs['locations_y'],
        np.asarray([100.0, 100.0, 100.0, 200.0, 200, 0.0]) / 500.0,
    )


if __name__ == '__main__':
  test_utils.main()
