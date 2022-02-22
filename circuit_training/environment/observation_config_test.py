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
"""Tests for circuit_training.environment.observation_config."""

from circuit_training.environment import observation_config
from circuit_training.utils import test_utils


class ObservationConfigTest(test_utils.TestCase):

  def test_flatten_unflatten(self):
    config = observation_config.ObservationConfig()
    obs = config.observation_space.sample()

    flatten_static_obs = observation_config.flatten_static(obs)
    self.assertLen(flatten_static_obs.shape, 1)
    static_tf_obs = observation_config.to_dict_static(flatten_static_obs)
    np_obs = {k: static_tf_obs[k].numpy() for k in static_tf_obs}
    for k in np_obs:
      self.assertAllEqual(obs[k], np_obs[k])

    flatten_dynamic_obs = observation_config.flatten_dynamic(obs)
    dynamic_tf_obs = observation_config.to_dict_dynamic(flatten_dynamic_obs)
    np_obs = {k: dynamic_tf_obs[k].numpy() for k in dynamic_tf_obs}
    for k in np_obs:
      self.assertAllEqual(obs[k], np_obs[k])

    flatten_all_obs = observation_config.flatten_all(obs)
    all_tf_obs = observation_config.to_dict_all(flatten_all_obs)
    np_obs = {k: all_tf_obs[k].numpy() for k in all_tf_obs}
    for k in np_obs:
      self.assertAllEqual(obs[k], np_obs[k])

  def test_observation_ordering(self):
    static_observations = (
        'normalized_num_edges',
        'normalized_num_hard_macros',
        'normalized_num_soft_macros',
        'normalized_num_port_clusters',
        'horizontal_routes_per_micron',
        'vertical_routes_per_micron',
        'macro_horizontal_routing_allocation',
        'macro_vertical_routing_allocation',
        'grid_cols',
        'grid_rows',
        'sparse_adj_i',
        'sparse_adj_j',
        'sparse_adj_weight',
        'edge_counts',
        'macros_w',
        'macros_h',
        'node_types',
    )

    dynamic_observations = (
        'locations_x',
        'locations_y',
        'is_node_placed',
        'current_node',
        'mask',
    )

    all_observations = static_observations + dynamic_observations

    # Make sure iterating order is only changed when we are deliberately
    # modifying the keys in the feature set. The relative ordering is important
    # because flatten/unflattening to/from a tensor is done by tf.split(). If
    # ordering is different, the state will be not encoded the same way across
    # training experiments/ evaluation runs.
    for expected, actual in zip(static_observations,
                                observation_config.STATIC_OBSERVATIONS):
      self.assertEqual(expected, actual)
    for expected, actual in zip(dynamic_observations,
                                observation_config.DYNAMIC_OBSERVATIONS):
      self.assertEqual(expected, actual)
    for expected, actual in zip(all_observations,
                                observation_config.ALL_OBSERVATIONS):
      self.assertEqual(expected, actual)


if __name__ == '__main__':
  test_utils.main()
