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
"""Coordinate descent placer library."""

import os
import time
from typing import Callable, Dict, Optional, List, Text, Tuple

from absl import logging
from circuit_training.environment import placement_util
from circuit_training.environment import plc_client
import numpy as np

NS_ORIENTATIONS = ['N', 'FN', 'S', 'FS']
EW_ORIENTATIONS = ['E', 'FE', 'W', 'FW']


class CoordinateDescentPlacer(object):
  """Coordinate descent algorithm to place nodes."""

  def __init__(self,
               plc: plc_client.PlacementCost,
               cost_fn: Callable[[plc_client.PlacementCost],
                                 Tuple[float, Dict[Text, float]]],
               epochs: int = 10,
               use_stdcell_placer: bool = False,
               stdcell_placer: Text = 'fd',
               node_order: Text = 'random',
               accept_bad_stdcell_moves: bool = False,
               stdcell_place_every_n_macros: int = 10,
               optimize_only_orientation: bool = False,
               cell_search_prob: float = 1.0,
               k_distance_bounded_search: bool = True,
               k_distance_bound: Optional[int] = None) -> None:
    """Creates a CoordinateDescentPlacer.

    Args:
      plc: The placement cost object.
      cost_fn: The cost function that gets the plc and returns cost and info.
      epochs: Number of epochs (iterations) in coordinate descend algorithm.
      use_stdcell_placer: If True, places stdcells using stdcell placer.
      stdcell_placer: Standad cell placer.
      node_order: Order of nodes to place using coordinate descent. Choose
        random, descending_size_macro_first, random_macro_first.
      accept_bad_stdcell_moves: If True, accept stdcell moves even if it leads
        to a higher cost.
      stdcell_place_every_n_macros: Run stdcell placement for every n macros. If
        None, run stdcell placement once after all macros are placed.
      optimize_only_orientation: If True, only search for best orientation of
        the hard macros.
      cell_search_prob: The probability to include a neighborhood cell to
        search. When it is 1.0, descents at the steepest direction.'
      k_distance_bounded_search: If True, only search best locations within k
        grid distance from current placed location. Does not apply to FD stdcell
        placer.
      k_distance_bound: If k_distance_bounded_search is True, only search within
        a neighborhood of at most k_distance_bound grid distance. If not
        spesified, it is set to max(cols, rows) // 3.
    """
    self.plc = plc
    self.cost_fn = cost_fn
    self._epochs = epochs
    self._node_order = node_order
    self._stdcell_place_every_n_macros = stdcell_place_every_n_macros
    self._cell_search_prob = cell_search_prob
    self._cols, self._rows = self.plc.get_grid_num_columns_rows()
    self._k_distance_bound = k_distance_bound or max(self._cols,
                                                     self._rows) // 3
    self._use_stdcell_placer = use_stdcell_placer
    self._stdcell_placer = stdcell_placer
    self._accept_bad_stdcell_moves = accept_bad_stdcell_moves
    self._optimize_only_orientation = optimize_only_orientation
    self._k_distance_bounded_search = k_distance_bounded_search

    if self._cell_search_prob < 0 or self._cell_search_prob > 1:
      raise ValueError(f'{self._cell_search_prob} should be between 0 and 1.')

    # Turn off incremental cost calculation if placing stdcells.
    if self._use_stdcell_placer:
      plc.set_use_incremental_cost(False)

    # Get legal node orientations.
    self._node_to_ori = {}
    for node in self.plc.get_macro_indices():
      if not self.plc.is_node_soft_macro(node):
        # TODO(wenjiej): Find orientation when a node is not placed initially.
        # Needed only when running CD from an empty grid.
        assert self.plc.is_node_placed(node)
        cur_ori = self.plc.get_macro_orientation(node)
        if cur_ori in NS_ORIENTATIONS:
          self._node_to_ori[node] = NS_ORIENTATIONS
        elif cur_ori in EW_ORIENTATIONS:
          self._node_to_ori[node] = EW_ORIENTATIONS
        else:
          raise ValueError(f'Unexpected orientation {cur_ori} for node {node}.')

    if self._use_stdcell_placer:
      plc.allow_hard_macros_over_std_cells(True)

    # If node order is random, will shuffle node orders for each iteration.
    self._ordered_node_indices = placement_util.get_ordered_node_indices(
        self._node_order, self.plc)

    # Exclude fixed macros with pre-determined locations.
    self._ordered_node_indices = [
        m for m in self._ordered_node_indices if not self.plc.is_node_fixed(m)
    ]

    self._soft_macro_indices = [
        m for m in self._ordered_node_indices if self.plc.is_node_soft_macro(m)
    ]

    if self._use_stdcell_placer:
      # Only include hard macros in self._ordered_node_indices.
      self._ordered_node_indices = [
          i for i in self._ordered_node_indices
          if not self.plc.is_node_soft_macro(i)
      ]
    logging.info('Total number of ordered nodes: %d',
                 len(self._ordered_node_indices))
    logging.info('ordered_node_indices: %s', self._ordered_node_indices)
    logging.info('Cost of initial placement: %s', self.report_cost())

  def find_best_location(self, node: int, mask: List[int],
                         locations: List[int]) -> Optional[int]:
    """Given a soft macro, search the best location."""
    best_loc = None
    best_cost = float('inf')
    for loc in locations:
      assert mask[loc] == 1
      self.plc.place_node(node, loc)
      new_cost, _ = self.cost_fn(self.plc)
      self.plc.unplace_node(node)
      if new_cost < best_cost:
        best_loc = loc
        best_cost = new_cost

    return best_loc

  def find_best_location_orientation(
      self, node: int, locations: List[int],
      orientations: List[Text]) -> Tuple[Optional[int], Optional[Text]]:
    """Given a hard macro, search the best location and orientation."""
    assert orientations
    best_loc = None
    best_ori = None
    best_cost = float('inf')

    for loc in locations:
      for ori in orientations:
        self.plc.place_node(node, loc)
        self.plc.update_macro_orientation(node, ori)
        new_cost, _ = self.cost_fn(self.plc)
        self.plc.unplace_node(node)
        if new_cost < best_cost:
          best_loc = loc
          best_ori = ori
          best_cost = new_cost

    return best_loc, best_ori

  def find_best_orientation(self, node: int,
                            orientations: List[Text]) -> Optional[Text]:
    """Given a hard macro, search the best orientation."""
    assert orientations
    best_ori = None
    best_cost = float('inf')

    for ori in orientations:
      self.plc.update_macro_orientation(node, ori)
      new_cost, _ = self.cost_fn(self.plc)
      if new_cost < best_cost:
        best_ori = ori
        best_cost = new_cost

    return best_ori

  def _get_row_col_from_cell(self, cell: int) -> Tuple[int, int]:
    return cell // self._cols, cell % self._cols

  def _get_cell_from_row_col(self, row: int, col: int) -> int:
    return int(row * self._cols + col)

  def _k_distance_bounded_locations(self, curr: int, k: int,
                                    locations: List[int]) -> List[int]:
    """Find k grid distance bounded locations from current cell."""
    curr_row, curr_col = self._get_row_col_from_cell(curr)
    bounded = []
    for c in locations:
      if c == curr:
        # Always include current location to search.
        bounded.append(c)
        continue

      row, col = self._get_row_col_from_cell(c)
      if abs(row - curr_row) + abs(col - curr_col) <= k:
        if np.random.random() <= self._cell_search_prob:
          bounded.append(c)
    return bounded

  def place_node(self, node: int) -> None:
    """Given a node, greedily place the node on the best location wrt cost."""
    if not self.plc.is_node_soft_macro(node):
      orientations = self._node_to_ori[node]

    if self._optimize_only_orientation:
      # Placing and unplacing macros cause wiered problems in FD.
      # See cl/316830807. Avoid unplacing for orientation optimization.
      best_ori = self.find_best_orientation(node, orientations)
      self.plc.update_macro_orientation(node, best_ori)
      return

    # Unplace the node from its current location to prepare placing node.
    curr_cell = self.plc.get_grid_cell_of_node(node)
    self.plc.unplace_node(node)

    mask = self.plc.get_node_mask(node)
    locations = [i for i, m in enumerate(mask) if m > 0]
    if not locations:
      # FD or DP are run between macro moves (_stdcell_place_every_n_macros).
      # They may place stdcells in a way that invalidates prior macro locations.
      # Stay with previous macro locations in this case.
      locations = [curr_cell]
      logging.info(
          'Cannot find feasible locations for node %d. '
          'Use its current location %d.', node, curr_cell)

    if self._k_distance_bounded_search:
      k = self._k_distance_bound
      # Increase search scope until there is at least one feasible location.
      while True:
        bounded = self._k_distance_bounded_locations(curr_cell, k, locations)
        if bounded:
          locations = bounded
          break
        else:
          k += self._k_distance_bound

    if self.plc.is_node_soft_macro(node):
      best_loc = self.find_best_location(node, mask, locations)
      self.plc.place_node(node, best_loc)
    else:
      best_loc, best_ori = self.find_best_location_orientation(
          node, locations, orientations)
      self.plc.place_node(node, best_loc)
      self.plc.update_macro_orientation(node, best_ori)

  def place_stdcells(self) -> None:
    """Place stdcells."""

    logging.info('Place stdcells using %s', self._stdcell_placer)
    old_cost, _ = self.cost_fn(self.plc)
    old_coordinates = [
        self.plc.get_node_location(m) for m in self._soft_macro_indices
    ]

    if self._stdcell_placer == 'fd':
      # Use default FD schedule.
      # Use current stdcell location to incrementally change stdcell locations
      # between iterations.
      placement_util.fd_placement_schedule(self.plc, use_current_loc=True)
    else:
      raise ValueError(
          f'stdcell placer {self._stdcell_placer} is not supported')

    new_cost, _ = self.cost_fn(self.plc)

    if new_cost > old_cost and not self._accept_bad_stdcell_moves:
      logging.info('Bad stdcell placement moves not accepted.')
      # Revert to old node coordinates.
      for i, (x, y) in enumerate(old_coordinates):
        self.plc.update_node_coords(self._soft_macro_indices[i], x, y)

  def optimize(self, epoch: int) -> None:
    """Performs one iteration (epoch) of coordinate descent on all nodes."""
    logging.info('Starts optimization in epoch %d.', epoch)
    start_time = time.time()

    node_indices = self._ordered_node_indices
    if self._node_order == 'random':
      np.random.shuffle(node_indices)

    for i, node in enumerate(node_indices):
      if i % 25 == 0:
        logging.info('Number of nodes placed by CD: %d', i)
      self.place_node(node)
      if (self._use_stdcell_placer and self._stdcell_place_every_n_macros and
          (i + 1) % self._stdcell_place_every_n_macros == 0):
        self.place_stdcells()

    # Always run stdcell placement after all macros are placed.
    if self._use_stdcell_placer:
      self.place_stdcells()

    logging.info('One iteration of coordinate descent takes %f seconds.',
                 (time.time() - start_time))

  def report_cost(self) -> Text:
    proxy_cost, info = self.cost_fn(self.plc)
    wirelength = info['wirelength']
    congestion = info['congestion']
    density = info['density']
    return ('(Objective cost, wirelength, congestion, density): ' +
            '({:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(proxy_cost, wirelength,
                                                     congestion, density))

  def place(self) -> None:
    """Place all nodes using coordinate descent algorithm for some iterations."""
    # Run stdcell placement at the beginning of the optimization loop if needed.
    # Use stdcell locations from initial placement.
    if self._use_stdcell_placer:
      self.place_stdcells()

    prev_cost, _ = self.cost_fn(self.plc)
    for i in range(self._epochs):
      self.optimize(i)
      logging.info('Cost after %d epochs: %s', i + 1, self.report_cost())
      curr_cost, _ = self.cost_fn(self.plc)
      if (prev_cost - curr_cost) / prev_cost < 1e-3:
        break
      prev_cost = curr_cost

  def save_placement(self, output_dir: Text, plc_filename: Text) -> None:
    """Saves a placement with current plc."""
    proxy_cost, info = self.cost_fn(self.plc)
    wirelength = info['wirelength']
    congestion = info['congestion']
    density = info['density']
    plc_filename_with_cost = 'cost_{:.4f}_w_{:.4f}_c_{:.4f}_d_{:.4f}_{}'.format(
        proxy_cost, wirelength, congestion, density, plc_filename)
    output_plc_file = os.path.join(output_dir, plc_filename_with_cost)
    placement_util.save_placement(self.plc, output_plc_file)
    # TODO(wenjiej): Enable saving plc view.
    # placement_util.save_as_svg(self.plc, f'{output_plc_file}.svg')
