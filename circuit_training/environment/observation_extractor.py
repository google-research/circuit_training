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

import collections
import copy
import math
from typing import Dict, Optional, Text, Tuple

from absl import logging
from circuit_training.environment import observation_config as observation_config_lib
from circuit_training.environment import plc_client
import gin
import numpy as np


@gin.configurable
class ObservationExtractor(object):
  """Extracts observation features from plc."""

  EPSILON = 1e-6
  ROUND_DIGITS = 4
  # Selected locations have at most this much L1 distance from the same size
  # macros. The distance is measured at number of aligned locations from the
  # placed macros with the same size..
  PACKING_MAX_DISTANCE = 5

  @staticmethod
  def _round_macro_size(macro_size: tuple[float, float]) -> tuple[float, float]:
    return (
        round(macro_size[0], ObservationExtractor.ROUND_DIGITS),
        round(macro_size[1], ObservationExtractor.ROUND_DIGITS),
    )

  def __init__(
      self,
      plc: plc_client.PlacementCost,
      observation_config: Optional[
          observation_config_lib.ObservationConfig
      ] = None,
      netlist_index: int = 0,
      default_location_x: float = 0.5,
      default_location_y: float = 0.5,
      use_plc_init_location: bool = True,
  ):
    self.plc = plc
    self._observation_config = (
        observation_config or observation_config_lib.ObservationConfig()
    )
    self._netlist_index = netlist_index
    self._default_location_x = default_location_x
    self._default_location_y = default_location_y
    self._use_plc_init_location = use_plc_init_location

    self.width, self.height = self.plc.get_canvas_width_height()
    self.half_perimeter = self.width + self.height
    self.num_cols, self.num_rows = self.plc.get_grid_num_columns_rows()
    self.grid_width = self.width / self.num_cols
    self.grid_height = self.height / self.num_rows

    # Padding for mapping the placement canvas on the agent canvas.
    rows_pad = self._observation_config.max_grid_size - self.num_rows
    cols_pad = self._observation_config.max_grid_size - self.num_cols
    self.up_pad = rows_pad // 2
    self.right_pad = cols_pad // 2
    self.low_pad = rows_pad - self.up_pad
    self.left_pad = cols_pad - self.right_pad

    # Since there are too many I/O ports, we have to cluster them together to
    # make it manageable for the model to process. The ports that are located in
    # the same grid cell are clustered togheter.
    self.adj_vec, grid_cell_of_clustered_ports_vec = (
        self.plc.get_macro_and_clustered_port_adjacency()
    )
    self.clustered_port_locations_vec = [
        self._get_clustered_port_locations(i)
        for i in grid_cell_of_clustered_ports_vec
    ]

    self.fake_net_dict = collections.defaultdict(dict)
    for fake_net in self.plc.get_fake_nets():
      weight = fake_net[0]
      node_0 = fake_net[1][0]
      node_1 = fake_net[1][1]
      self.fake_net_dict[node_0][node_1] = weight
      self.fake_net_dict[node_1][node_0] = weight

    hard_macro_indices = [
        m
        for m in self.plc.get_macro_indices()
        if not self.plc.is_node_soft_macro(m)
    ]
    self._macro_size_dict = collections.defaultdict(list)
    for m in hard_macro_indices:
      w, h = self.plc.get_node_width_height(m)
      self._macro_size_dict[
          ObservationExtractor._round_macro_size((w, h))
      ].append(m)

    # Extract static features.
    self._initial_features = self._extract_static_features()
    self.reset()

  def reset(self) -> None:
    self._features = copy.deepcopy(self._initial_features)

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
    self._extract_netlist_index(features)
    self._extract_normalized_static_features(features)
    return features

  def _extract_normalized_static_features(
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    """Normalizes static features."""
    self._add_netlist_metadata(features)
    self._normalize_adj_matrix(features)
    self._pad_adj_matrix(features)
    self._pad_macro_static_features(features)
    self._normalize_macro_size_by_canvas(features)
    self._normalize_canvas_size_by_canvas(features)
    self._normalize_grid_size(features)
    self._normalize_locations_by_canvas(features)
    self._pad_macro_dynamic_features(features)

  def _extract_num_macros(self, features: Dict[Text, np.ndarray]) -> None:
    features['num_macros'] = np.asarray(
        [len(self.plc.get_macro_indices())]
    ).astype(np.int32)

  def _extract_technology_info(self, features: Dict[Text, np.ndarray]) -> None:
    """Extracts Technology-related information."""
    # Divide these by 1000.0 to make their values closer to the rest of the
    # features.
    routing_resources = {
        'total_horizontal_routes_k': (
            self.plc.get_routes_per_micron()[0] * self.height / 1000.0
        ),
        'total_vertical_routes_k': (
            self.plc.get_routes_per_micron()[1] * self.width / 1000.0
        ),
        'total_macro_horizontal_routes_k': (
            self.plc.get_macro_routing_allocation()[0] * self.height / 1000.0
        ),
        'total_macro_vertical_routes_k': (
            self.plc.get_macro_routing_allocation()[0] * self.width / 1000.0
        ),
    }
    for k in routing_resources:
      features[k] = np.asarray([routing_resources[k]]).astype(np.float32)

  def _extract_initial_node_locations(
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    """Extracts initial node locations."""
    # We read the initial locations from the plc, but we set `is_node_placed` to
    # false for movable items, since they will be unplaced by the env after
    # initialization.
    locations_x = []
    locations_y = []
    is_node_placed = []
    for macro_idx in self.plc.get_macro_indices():
      if self.plc.is_node_fixed(macro_idx) or (
          self._use_plc_init_location and self.plc.is_node_placed(macro_idx)
      ):
        x, y = self.plc.get_node_location(macro_idx)
      else:
        x = self._default_location_x * self.width
        y = self._default_location_y * self.height
      locations_x.append(x)
      locations_y.append(y)
      is_node_placed.append(self.plc.is_node_fixed(macro_idx))
    for x, y in self.clustered_port_locations_vec:
      locations_x.append(x)
      locations_y.append(y)
      is_node_placed.append(True)
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
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    """Extracts adjacency matrix."""
    num_nodes = len(self.plc.get_macro_indices()) + len(
        self.clustered_port_locations_vec
    )
    assert num_nodes * num_nodes == len(self.adj_vec)
    sparse_adj_i = []
    sparse_adj_j = []
    sparse_adj_weight = []
    edge_counts = np.zeros(
        (self._observation_config.max_num_nodes,), dtype=np.int32
    )
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
        np.float32
    )
    features['edge_counts'] = edge_counts

  def _extract_canvas_size(self, features: Dict[Text, np.ndarray]) -> None:
    features['canvas_width'] = np.asarray([self.width]).astype(np.float32)
    features['canvas_height'] = np.asarray([self.height]).astype(np.float32)

  def _extract_grid_size(self, features: Dict[Text, np.ndarray]) -> None:
    features['grid_cols'] = np.asarray([self.num_cols]).astype(np.float32)
    features['grid_rows'] = np.asarray([self.num_rows]).astype(np.float32)

  def _extract_netlist_index(self, features: Dict[Text, np.ndarray]) -> None:
    features['netlist_index'] = np.asarray([self._netlist_index]).astype(
        np.int32
    )

  def _get_clustered_port_locations(
      self, grid_cell_index: int
  ) -> Tuple[float, float]:
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
        np.sum(features['sparse_adj_weight'])
        / self._observation_config.max_num_edges
    ]).astype(np.float32)
    features['normalized_num_hard_macros'] = np.asarray([
        np.sum(
            np.equal(
                features['node_types'], observation_config_lib.HARD_MACRO
            ).astype(np.float32)
        )
        / self._observation_config.max_num_nodes
    ]).astype(np.float32)
    features['normalized_num_soft_macros'] = np.asarray([
        np.sum(
            np.equal(
                features['node_types'], observation_config_lib.SOFT_MACRO
            ).astype(np.float32)
        )
        / self._observation_config.max_num_nodes
    ]).astype(np.float32)
    features['normalized_num_port_clusters'] = np.asarray([
        np.sum(
            np.equal(
                features['node_types'], observation_config_lib.PORT_CLUSTER
            ).astype(np.float32)
        )
        / self._observation_config.max_num_nodes
    ]).astype(np.float32)

  def _normalize_adj_matrix(self, features: Dict[Text, np.ndarray]) -> None:
    """Normalizes adj matrix weights."""
    mean_weight = np.mean(features['sparse_adj_weight'])
    features['sparse_adj_weight'] = (
        features['sparse_adj_weight']
        / (mean_weight + ObservationExtractor.EPSILON)
    ).astype(np.float32)

  def _pad_1d_tensor(self, tensor: np.ndarray, pad_size: int) -> np.ndarray:
    return np.pad(
        tensor,
        (0, pad_size - tensor.shape[0]),
        mode='constant',
        constant_values=0,
    )

  def _pad_adj_matrix(self, features: Dict[Text, np.ndarray]) -> None:
    """Pads indices and weights with zero to make their shape known."""
    logging.info(
        'Pad a tensor with shape %s by %s',
        features['sparse_adj_i'].shape,
        self._observation_config.max_num_edges,
    )
    for var in ['sparse_adj_i', 'sparse_adj_j', 'sparse_adj_weight']:
      features[var] = self._pad_1d_tensor(
          features[var], self._observation_config.max_num_edges
      )

  def _pad_macro_static_features(
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    """Pads macro features to make their shape knwon."""
    logging.info(
        'Pad a tensor with shape %s by %s',
        features['macros_w'].shape,
        self._observation_config.max_num_nodes,
    )
    for var in [
        'macros_w',
        'macros_h',
        'node_types',
    ]:
      features[var] = self._pad_1d_tensor(
          features[var], self._observation_config.max_num_nodes
      )

  def _pad_macro_dynamic_features(
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    """Pads macro features to make their shape knwon."""
    for var in [
        'locations_x',
        'locations_y',
        'is_node_placed',
    ]:
      features[var] = self._pad_1d_tensor(
          features[var], self._observation_config.max_num_nodes
      )

  def _normalize_canvas_size_by_canvas(
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    features['canvas_width'] = (
        features['canvas_width'] / (self.half_perimeter)
    ).astype(np.float32)
    features['canvas_height'] = (
        features['canvas_height'] / (self.half_perimeter)
    ).astype(np.float32)

  def _normalize_grid_size(self, features: Dict[Text, np.ndarray]) -> None:
    features['grid_cols'] = (
        features['grid_cols'] / self._observation_config.max_grid_size
    ).astype(np.float32)
    features['grid_rows'] = (
        features['grid_rows'] / self._observation_config.max_grid_size
    ).astype(np.float32)

  def _normalize_macro_size_by_canvas(
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    """Normalizes macro sizes with the canvas size."""
    features['macros_w'] = (
        features['macros_w'] / (self.half_perimeter)
    ).astype(np.float32)
    features['macros_h'] = (
        features['macros_h'] / (self.half_perimeter)
    ).astype(np.float32)

  def _normalize_locations_by_canvas(
      self, features: Dict[Text, np.ndarray]
  ) -> None:
    """Normalizes locations with the canvas size."""
    features['locations_x'] = (
        features['locations_x'] / self.half_perimeter
    ).astype(np.float32)
    features['locations_y'] = (
        features['locations_y'] / self.half_perimeter
    ).astype(np.float32)

  def _grid_l2_distance(self, loc1: np.ndarray, loc2: np.ndarray) -> np.ndarray:
    """Returns the l2 distance between loc1 and loc2.

    Args:
      loc1: First grid id.
      loc2: Second grid id.

    Returns:
      The L2 distance between the grid locations.
    """
    r1 = loc1 // self.num_cols
    c1 = loc1 % self.num_cols

    r2 = loc2 // self.num_cols
    c2 = loc2 % self.num_cols

    return np.sqrt(
        ((c1 - c2) * self.grid_width) ** 2 + ((r1 - r2) * self.grid_height) ** 2
    )

  def get_fake_net_heatmap(self, current_node_index: int) -> np.ndarray:
    """Returns the fake net heatmap for the current node.

    The heatmap is generated by calculating the additional fake net cost for
    placing the current macro on each location of the placement grid.

    Args:
      current_node_index: Index to the current node to be placed.

    Returns:
      A heatmap image with the size of max_grid_size**2 for fake net cost.
    """

    zero_heatmap = np.zeros(
        (self._observation_config.max_grid_size**2,), dtype=np.float32
    )
    if current_node_index < 0 or current_node_index >= len(
        self.plc.get_macro_indices()
    ):
      return zero_heatmap
    current_node_id = self.plc.get_macro_indices()[current_node_index]
    if current_node_id not in self.fake_net_dict:
      return zero_heatmap
    related_fake_nets = self.fake_net_dict[current_node_id]

    all_locations = np.arange(self.num_rows * self.num_cols)

    placed_nodes = [
        node for node in related_fake_nets if self.plc.is_node_placed(node)
    ]
    related_node_locs = np.array(
        [self.plc.get_grid_cell_of_node(node) for node in placed_nodes]
    )
    weights = np.array([related_fake_nets[node] for node in placed_nodes])
    costs = np.sum(
        weights[:, np.newaxis]
        * self._grid_l2_distance(
            related_node_locs[:, np.newaxis], all_locations
        ),
        axis=0,
    )

    # Use negative cost, so the smallest cost becomes 1.0 after normalization.
    heatmap = -costs.reshape(self.num_rows, self.num_cols)

    # Normalize between 0 and 1.
    h_min = np.min(heatmap)
    h_max = np.max(heatmap)
    normalized_heatmap = (heatmap - h_min) / (
        h_max - h_min + ObservationExtractor.EPSILON
    )

    # Pad the image to the max_grid_size.
    pad = ((self.up_pad, self.low_pad), (self.right_pad, self.left_pad))
    normalized_heatmap = np.pad(
        normalized_heatmap, pad, mode='constant', constant_values=0.0
    )
    return np.reshape(
        normalized_heatmap, (self._observation_config.max_grid_size**2,)
    )

  def get_packing_heatmap(self, current_node_index: int) -> np.ndarray:
    """Returns the packing heatmap for the current node.

    The heatmap is non-zero for locations that are aligned with the already
    placed macro with the same size. The heatmap has a larger value for
    locations that are closer to the placed macros of the same size. The value
    decreases 1/l1-distance from the placed macros where the distance is
    measured by how many aligned macros can be placed between the two locations.
    We don't set the heatmap in the location with distance 0 (where the placed
    macro is located because that location is not available.)

    Args:
      current_node_index: The index of the current node.

    Returns:
      The packing heatmap.
    """
    if current_node_index < 0 or current_node_index >= len(
        self.plc.get_macro_indices()
    ):
      return np.zeros(
          (self._observation_config.max_grid_size**2,), dtype=np.float32
      )

    current_node_id = self.plc.get_macro_indices()[current_node_index]
    w, h = self.plc.get_node_width_height(current_node_id)

    macro_grid_size = (
        math.ceil(w / self.grid_width),
        math.ceil(h / self.grid_height),
    )

    same_size_macros = self._macro_size_dict[
        ObservationExtractor._round_macro_size((w, h))
    ]
    heatmap = np.zeros((self.num_rows, self.num_cols), dtype=np.float32)
    for m in same_size_macros:
      if self.plc.is_node_placed(m):
        grid_id = self.plc.get_grid_cell_of_node(m)
        r = grid_id // self.num_cols
        c = grid_id % self.num_cols
        max_d = ObservationExtractor.PACKING_MAX_DISTANCE
        for i in range(-max_d, max_d + 1):
          for j in range(-max_d, max_d + 1):
            d = abs(i) + abs(j)
            if (
                0 < d <= max_d
                and 0 <= r + i * macro_grid_size[1] < self.num_rows
                and 0 <= c + j * macro_grid_size[0] < self.num_cols
            ):
              heatmap[
                  r + i * macro_grid_size[1], c + j * macro_grid_size[0]
              ] += (1.0 / d)

    # Normalize between 0 and 1.
    h_max = np.max(heatmap)
    normalized_heatmap = (heatmap) / (h_max + ObservationExtractor.EPSILON)

    pad = ((self.up_pad, self.low_pad), (self.right_pad, self.left_pad))
    normalized_heatmap = np.pad(
        normalized_heatmap, pad, mode='constant', constant_values=0.0
    )
    return np.reshape(
        normalized_heatmap, (self._observation_config.max_grid_size**2,)
    )

  def get_static_features(self) -> Dict[Text, np.ndarray]:
    return {
        key: self._features[key]
        for key in observation_config_lib.STATIC_OBSERVATIONS
    }

  def _update_dynamic_features(
      self, previous_node_index: int, current_node_index: int, mask: np.ndarray
  ) -> None:
    """Updates the dynamic features."""
    if previous_node_index >= 0:
      x, y = self.plc.get_node_location(
          self.plc.get_macro_indices()[previous_node_index]
      )
      self._features['locations_x'][previous_node_index] = (
          x / self.half_perimeter
      )
      self._features['locations_y'][previous_node_index] = (
          y / self.half_perimeter
      )
      self._features['is_node_placed'][previous_node_index] = 1
    self._features['fake_net_heatmap'] = self.get_fake_net_heatmap(
        current_node_index
    )
    self._features['packing_heatmap'] = self.get_packing_heatmap(
        current_node_index
    )
    self._features['mask'] = mask.astype(np.int32)
    self._features['current_node'] = np.asarray([current_node_index]).astype(
        np.int32
    )

  def get_dynamic_features(
      self, previous_node_index: int, current_node_index: int, mask: np.ndarray
  ) -> Dict[Text, np.ndarray]:
    self._update_dynamic_features(previous_node_index, current_node_index, mask)
    return {
        key: copy.deepcopy(self._features[key])
        for key in observation_config_lib.DYNAMIC_OBSERVATIONS
        if key in self._features
    }

  def get_all_features(
      self, previous_node_index: int, current_node_index: int, mask: np.ndarray
  ) -> Dict[Text, np.ndarray]:
    features = self.get_static_features()
    features.update(
        self.get_dynamic_features(
            previous_node_index=previous_node_index,
            current_node_index=current_node_index,
            mask=mask,
        )
    )
    return features
