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
"""Tests for static_feature_cache."""

from circuit_training.environment import observation_config
from circuit_training.learning import static_feature_cache
import numpy as np
from tf_agents.utils import test_utils


class StaticFeatureCacheTest(test_utils.TestCase):

  def test_no_add_empty_features(self):
    cache = static_feature_cache.StaticFeatureCache()
    all_static_features = cache.get_all_static_features()
    for feature in observation_config.STATIC_OBSERVATIONS:
      np.testing.assert_equal(all_static_features[feature], np.array([]))

  def test_add_two_features_expect_two_features(self):
    cache = static_feature_cache.StaticFeatureCache()
    obs_space = observation_config.ObservationConfig().observation_space
    sample1 = obs_space.sample()
    sample1['netlist_index'] = np.array([10])

    cache.add_static_feature(sample1)
    all_static_features = cache.get_all_static_features()
    np.testing.assert_equal(all_static_features['netlist_index'], [[10]])
    for feature in observation_config.STATIC_OBSERVATIONS:
      np.testing.assert_equal(all_static_features[feature],
                              np.array([sample1[feature]]))

    sample2 = obs_space.sample()
    sample2['netlist_index'] = np.array([8])
    cache.add_static_feature(sample2)
    all_static_features = cache.get_all_static_features()
    np.testing.assert_equal(all_static_features['netlist_index'], [[8], [10]])
    for feature in observation_config.STATIC_OBSERVATIONS:
      np.testing.assert_equal(all_static_features[feature],
                              np.array([sample2[feature], sample1[feature]]))


if __name__ == '__main__':
  test_utils.main()
