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
"""Handling static feature caching."""

from typing import Dict

from circuit_training.environment import observation_config
import numpy as np


class StaticFeatureCache:
  """A class to handle static feature caching."""

  def __init__(self):
    self.static_feature_dict = {}

  def add_static_feature(self, static_feature: Dict[str, np.ndarray]) -> None:
    netlist_index = static_feature['netlist_index'][0]
    if netlist_index in self.static_feature_dict:
      raise ValueError(
          'Adding two static features with '
          f'the same netlist_index: {netlist_index}.')
    self.static_feature_dict[netlist_index] = static_feature

  def get_all_static_features(self) -> Dict[str, np.ndarray]:
    """Returns the stacked static feature with netlist_index order."""
    netlist_indices = sorted(list(self.static_feature_dict.keys()))

    all_static_features = {}
    for feature in observation_config.STATIC_OBSERVATIONS:
      all_static_features[feature] = np.array([
          self.static_feature_dict[netlist_index][feature]
          for netlist_index in netlist_indices
      ])
    return all_static_features
