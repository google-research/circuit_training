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
"""This class extracts features from observations."""

from typing import Dict, Optional, Text, Tuple

from circuit_training.environment import observation_config as observation_config_lib
from circuit_training.environment import plc_client
import gin
import numpy as np


@gin.configurable
class ObservationExtractor(object):
  """Extracts observation features from plc."""

  EPSILON = 1E-6

  def __init__(self,
               plc: plc_client.PlacementCost,
               observation_config: Optional[
                   observation_config_lib.ObservationConfig] = None,
               default_location_x: float = 0.5,
               default_location_y: float = 0.5):
    self.plc = plc
    self._observation_config = (
        observation_config or observation_config_lib.ObservationConfig())
    self._default_location_x = default_location_x
    self._default_location_y = default_location_y

    self.width, self.height = self.plc.get_canvas_width_height()
    self.num_cols, self.num_rows = self.plc.get_grid_num_columns_rows()
    self.grid_width = self.width / self.num_cols
    self.grid_height = self.height / self.num_rows

    # Since there are too many I/O ports, we have to cluster them together to
    # make it manageable for the model to process. The ports that are located in
    # the same grid cell are clustered togheter.
    self.adj_vec, grid_cell_of_clustered_ports_vec = self.plc.get_macro_and_clustered_port_adjacency(
    )
    self.clustered_port_locations_vec = [
        self._get_clustered_port_locations(i)
        for i in grid_cell_of_clustered_ports_vec
    ]

    # Extract static features.
    self._features = self._extract_static_features()

  def _extract_static_features(self) -> Dict[Text, np.ndarray]:
    """Static features that are invariant across training steps."""
    features = dict()
    self._extract_num_macros(features)
    self._extract_technology_info(features)
    self._extract_node_types(features)
    self._extract_macro_size(features)
    self._extract_macro_and_port_adj_matrix(features)
    self._extract_canvas_size(features)
    self._extract_grid_size(features)
    self._extract_initial_node_locations(features)
    self._extract_normalized_static_features(features)
    return features

  def _extract_normalized_static_features(
      self, features: Dict[Text, np.ndarray]) -> None:
    """Normalizes static features."""
    self._add_netlist_metadata(features)
    self._normalize_adj_matrix(features)
    self._pad_adj_matrix(features)
    self._pad_macro_static_features(features)
    self._normalize_macro_size_by_canvas(features)
    self._normalize_grid_size(features)
    self._normalize_locations_by_canvas(features)
    self._replace_unplace_node_location(features)
    self._pad_macro_dynamic_features(features)

  def _extract_num_macros(self, features: Dict[Text, np.ndarray]) -> None:
    features['num_macros'] = np.asarray([len(self.plc.get_macro_indices())
                                        ]).astype(np.int32)

  def _extract_technology_info(self, features: Dict[Text, np.ndarray]) -> None:
    """Extracts Technology-related information."""
    routing_resources = {
        'horizontal_routes_per_micron':
            self.plc.get_routes_per_micron()[0],
        'vertical_routes_per_micron':
            self.plc.get_routes_per_micron()[1],
        'macro_horizontal_routing_allocation':
            self.plc.get_macro_routing_allocation()[0],
        'macro_vertical_routing_allocation':
            self.plc.get_macro_routing_allocation()[0],
    }
    for k in routing_resources:
      features[k] = np.asarray([routing_resources[k]]).astype(np.float32)

  def _extract_initial_node_locations(self, features: Dict[Text,
                                                           np.ndarray]) -> None:
    """Extracts initial node locations."""
    locations_x = []
    locations_y = []
    is_node_placed = []
    for macro_idx in self.plc.get_macro_indices():
      x, y = self.plc.get_node_location(macro_idx)
      locations_x.append(x)
      locations_y.append(y)
      is_node_placed.append(1 if self.plc.is_node_placed(macro_idx) else 0)
    for x, y in self.clustered_port_locations_vec:
      locations_x.append(x)
      locations_y.append(y)
      is_node_placed.append(1)
    features['locations_x'] = np.asarray(locations_x).astype(np.float32)
    features['locations_y'] = np.asarray(locations_y).astype(np.float32)
    features['is_node_placed'] = np.asarray(is_node_placed).astype(np.int32)

  def _extract_node_types(self, features: Dict[Text, np.ndarray]) -> None:
    """Extracts node types."""
    types = []
    for macro_idx in self.plc.get_macro_indices():
      if self.plc.is_node_soft_macro(macro_idx):
        types.append(observation_config_lib.SOFT_MACRO)
      else:
        types.append(observation_config_lib.HARD_MACRO)
    for _ in range(len(self.clustered_port_locations_vec)):
      types.append(observation_config_lib.PORT_CLUSTER)
    features['node_types'] = np.asarray(types).astype(np.int32)

  def _extract_macro_size(self, features: Dict[Text, np.ndarray]) -> None:
    """Extracts macro sizes."""
    macros_w = []
    macros_h = []
    for macro_idx in self.plc.get_macro_indices():
      if self.plc.is_node_soft_macro(macro_idx):
        # Width and height of soft macros are set to zero.
        width = 0
        height = 0
      else:
        width, height = self.plc.get_node_width_height(macro_idx)
      macros_w.append(width)
      macros_h.append(height)
    for _ in range(len(self.clustered_port_locations_vec)):
      macros_w.append(0)
      macros_h.append(0)
    features['macros_w'] = np.asarray(macros_w).astype(np.float32)
    features['macros_h'] = np.asarray(macros_h).astype(np.float32)

  def _extract_macro_and_port_adj_matrix(
      self, features: Dict[Text, np.ndarray]) -> None:
    """Extracts adjacency matrix."""
    num_nodes = len(self.plc.get_macro_indices()) + len(
        self.clustered_port_locations_vec)
    assert num_nodes * num_nodes == len(self.adj_vec)
    sparse_adj_i = []
    sparse_adj_j = []
    sparse_adj_weight = []
    edge_counts = np.zeros((self._observation_config.max_num_nodes,),
                           dtype=np.int32)
    for i in range(num_nodes):
      for j in range(i + 1, num_nodes):
        weight = self.adj_vec[i + num_nodes * j]
        if weight > 0:
          sparse_adj_i.append(i)
          sparse_adj_j.append(j)
          sparse_adj_weight.append(weight)
          edge_counts[i] += 1
          edge_counts[j] += 1

    features['sparse_adj_i'] = np.asarray(sparse_adj_i).astype(np.int32)
    features['sparse_adj_j'] = np.asarray(sparse_adj_j).astype(np.int32)
    features['sparse_adj_weight'] = np.asarray(sparse_adj_weight).astype(
        np.float32)
    features['edge_counts'] = edge_counts

  def _extract_canvas_size(self, features: Dict[Text, np.ndarray]) -> None:
    features['canvas_width'] = np.asarray([self.width])
    features['canvas_height'] = np.asarray([self.height])

  def _extract_grid_size(self, features: Dict[Text, np.ndarray]) -> None:
    features['grid_cols'] = np.asarray([self.num_cols]).astype(np.float32)
    features['grid_rows'] = np.asarray([self.num_rows]).astype(np.float32)

  def _get_clustered_port_locations(
      self, grid_cell_index: int) -> Tuple[float, float]:
    """Returns clustered port locations.

    This function returns an approximation location of the ports in a grid
    cell. Depending on the cell location in the canvas, the approximation
    differs.

    Args:
      grid_cell_index: The index of the grid cell where the cluster port is
        located.

    Returns:
      A tuple of float: Approximate x, y location of the port cluster in the
      grid cell in the same unit as canvas width and height (micron).
    """
    col = grid_cell_index % self.num_cols
    row = grid_cell_index // self.num_cols
    if col == 0 and row == 0:
      return 0, 0
    elif col == 0 and row == self.num_rows - 1:
      return 0, self.height
    elif col == self.num_cols - 1 and row == 0:
      return self.width, 0
    elif col == self.num_cols - 1 and row == self.num_rows - 1:
      return self.width, self.height
    elif col == 0:
      return 0, (row + 0.5) * self.grid_height
    elif col == self.num_cols - 1:
      return self.width, (row + 0.5) * self.grid_height
    elif row == 0:
      return (col + 0.5) * self.grid_width, 0
    elif row == self.num_rows - 1:
      return (col + 0.5) * self.grid_width, self.height
    else:
      return (col + 0.5) * self.grid_width, (row + 0.5) * self.grid_height

  def _add_netlist_metadata(self, features: Dict[Text, np.ndarray]) -> None:
    """Adds netlist metadata info."""
    features['normalized_num_edges'] = np.asarray([
        np.sum(features['sparse_adj_weight']) /
        self._observation_config.max_num_edges
    ]).astype(np.float32)
    features['normalized_num_hard_macros'] = np.asarray([
        np.sum(
            np.equal(features['node_types'],
                     observation_config_lib.HARD_MACRO).astype(np.float32)) /
        self._observation_config.max_num_nodes
    ]).astype(np.float32)
    features['normalized_num_soft_macros'] = np.asarray([
        np.sum(
            np.equal(features['node_types'],
                     observation_config_lib.SOFT_MACRO).astype(np.float32)) /
        self._observation_config.max_num_nodes
    ]).astype(np.float32)
    features['normalized_num_port_clusters'] = np.asarray([
        np.sum(
            np.equal(features['node_types'],
                     observation_config_lib.PORT_CLUSTER).astype(np.float32)) /
        self._observation_config.max_num_nodes
    ]).astype(np.float32)

  def _normalize_adj_matrix(self, features: Dict[Text, np.ndarray]) -> None:
    """Normalizes adj matrix weights."""
    mean_weight = np.mean(features['sparse_adj_weight'])
    features['sparse_adj_weight'] = (
        features['sparse_adj_weight'] /
        (mean_weight + ObservationExtractor.EPSILON)).astype(np.float32)

  def _pad_1d_tensor(self, tensor: np.ndarray, pad_size: int) -> np.ndarray:
    return np.pad(
        tensor, (0, pad_size - tensor.shape[0]),
        mode='constant',
        constant_values=0)

  def _pad_adj_matrix(self, features: Dict[Text, np.ndarray]) -> None:
    """Pads indices and weights with zero to make their shape known."""
    for var in ['sparse_adj_i', 'sparse_adj_j', 'sparse_adj_weight']:
      features[var] = self._pad_1d_tensor(
          features[var], self._observation_config.max_num_edges)

  def _pad_macro_static_features(self, features: Dict[Text,
                                                      np.ndarray]) -> None:
    """Pads macro features to make their shape knwon."""
    for var in [
        'macros_w',
        'macros_h',
        'node_types',
    ]:
      features[var] = self._pad_1d_tensor(
          features[var], self._observation_config.max_num_nodes)

  def _pad_macro_dynamic_features(self, features: Dict[Text,
                                                       np.ndarray]) -> None:
    """Pads macro features to make their shape knwon."""
    for var in [
        'locations_x',
        'locations_y',
        'is_node_placed',
    ]:
      features[var] = self._pad_1d_tensor(
          features[var], self._observation_config.max_num_nodes)

  def _normalize_grid_size(self, features: Dict[Text, np.ndarray]) -> None:
    features['grid_cols'] = (features['grid_cols'] /
                             self._observation_config.max_grid_size).astype(
                                 np.float32)
    features['grid_rows'] = (features['grid_rows'] /
                             self._observation_config.max_grid_size).astype(
                                 np.float32)

  def _normalize_macro_size_by_canvas(self, features: Dict[Text,
                                                           np.ndarray]) -> None:
    """Normalizes macro sizes with the canvas size."""
    features['macros_w'] = (
        features['macros_w'] /
        (features['canvas_width'] + ObservationExtractor.EPSILON)).astype(
            np.float32)
    features['macros_h'] = (
        features['macros_h'] /
        (features['canvas_height'] + ObservationExtractor.EPSILON)).astype(
            np.float32)

  def _normalize_locations_by_canvas(self, features: Dict[Text,
                                                          np.ndarray]) -> None:
    """Normalizes locations with the canvas size."""
    features['locations_x'] = (
        features['locations_x'] /
        (features['canvas_width'] + ObservationExtractor.EPSILON)).astype(
            np.float32)
    features['locations_y'] = (
        features['locations_y'] /
        (features['canvas_height'] + ObservationExtractor.EPSILON)).astype(
            np.float32)

  def _replace_unplace_node_location(self, features: Dict[Text,
                                                          np.ndarray]) -> None:
    """Replace the location of the unplaced macros with a constant."""
    is_node_placed = np.equal(features['is_node_placed'], 1)
    features['locations_x'] = np.where(
        is_node_placed,
        features['locations_x'],
        self._default_location_x * np.ones_like(features['locations_x']),
    ).astype(np.float32)
    features['locations_y'] = np.where(
        is_node_placed,
        features['locations_y'],
        self._default_location_y * np.ones_like(features['locations_y']),
    ).astype(np.float32)

  def get_static_features(self) -> Dict[Text, np.ndarray]:
    return {
        key: self._features[key]
        for key in observation_config_lib.STATIC_OBSERVATIONS
    }

  def get_initial_features(self) -> Dict[Text, np.ndarray]:
    return {
        key: self._features[key]
        for key in observation_config_lib.INITIAL_OBSERVATIONS
    }

  def _update_dynamic_features(self, previous_node_index: int,
                               current_node_index: int,
                               mask: np.ndarray) -> None:
    """Updates the dynamic features."""
    if previous_node_index >= 0:
      x, y = self.plc.get_node_location(
          self.plc.get_macro_indices()[previous_node_index])
      self._features['locations_x'][previous_node_index] = (
          x / (self.width + ObservationExtractor.EPSILON))
      self._features['locations_y'][previous_node_index] = (
          y / (self.height + ObservationExtractor.EPSILON))
      self._features['is_node_placed'][previous_node_index] = 1
    self._features['mask'] = mask.astype(np.int32)
    self._features['current_node'] = np.asarray([current_node_index
                                                ]).astype(np.int32)

  def get_dynamic_features(self, previous_node_index: int,
                           current_node_index: int,
                           mask: np.ndarray) -> Dict[Text, np.ndarray]:
    self._update_dynamic_features(previous_node_index, current_node_index, mask)
    return {
        key: self._features[key]
        for key in observation_config_lib.DYNAMIC_OBSERVATIONS
        if key in self._features
    }

  def get_all_features(self, previous_node_index: int, current_node_index: int,
                       mask: np.ndarray) -> Dict[Text, np.ndarray]:
    features = self.get_static_features()
    features.update(
        self.get_dynamic_features(
            previous_node_index=previous_node_index,
            current_node_index=current_node_index,
            mask=mask))
    return features
