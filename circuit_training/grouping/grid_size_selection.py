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
"""Utilities for choosing a good number of columns/rows for a given netlist.

Main function to call is get_grid_suggestion, it will return a single number
for columns and rows if there's a feasible solution with the flag settings.

Key function: get_grid_choices
It returns all possible col x row options with a list of metrics for empty
cells ratio (when all hard macros are placed), and wasted space ratios when
all cells are placed side by side (in horizintal direction) of on top of each
other (vertical direction) with the choice of grid cells width, and height.

The other important function that chooses one col x row choice from these
choices is: select_from_grid_choices
"""

import random
import dataclasses

from absl import flags
from circuit_training.environment import placement_util

flags.DEFINE_integer('min_num', 10, 'Minimum number for cols/rows sweep.')
flags.DEFINE_integer('max_num', 128, 'Maximum number for cols/rows sweep.')
flags.DEFINE_float('add_size', 0.0,
                   'Add to segment sizes to leave space between macros.')

# TODO(esonghori): Consider increasing when SA is not needed anymore.
flags.DEFINE_integer('max_num_grid_cells', 2500, 'max num of grid cells')
flags.DEFINE_integer('min_num_grid_cells', 500, 'min num of grid cells')
flags.DEFINE_float('max_aspect_ratio', 1.5,
                   'Maximum aspect ratio of a grid cell (either w/h of h/w)')
flags.register_validator('max_aspect_ratio', lambda x: x > 1.0)
# Tolerance helps to favor lower number of grid cells if the metric that choice
# is within this much compared to the best metric.
flags.DEFINE_float('tolerance', 0.05,
                   'Tolerance to choose lower number of grids')
flags.DEFINE_boolean('grid_select_include_fixed_macros', False, 'If set, '
                     'include fixed macro in grid selection.')
flags.DEFINE_integer(
    'max_grid_size', 128,
    'The maximum grid size of the canvas. The padded canvas '
    'will be max_grid_size by max_grid_size. '
    'Used only in the generalization model.')

FLAGS = flags.FLAGS


def get_grid_suggestion(plc):
  """Returns a single suggestion for number of columns and rows."""
  # Save previous placements, since get_grid_choices will move macros around
  # during evaluation. Note that this is in terms of absolute x, y coordinates,
  # not grid cell indices.
  orig_coords = placement_util.get_node_xy_coordinates(plc)
  orig_orientations = placement_util.get_macro_orientations(plc)
  choices = get_grid_choices(plc, FLAGS.min_num, FLAGS.max_num,
                             FLAGS.max_grid_size, FLAGS.min_num_grid_cells,
                             FLAGS.max_num_grid_cells, FLAGS.max_aspect_ratio,
                             FLAGS.add_size,
                             FLAGS.grid_select_include_fixed_macros)

  placement_util.restore_node_xy_coordinates(plc, orig_coords)
  placement_util.restore_macro_orientations(plc, orig_orientations)
  if not choices:
    return None, None
  cols, rows = select_from_grid_choices(
      choices, FLAGS.tolerance, print_best_n=20)
  return cols, rows


def get_span_and_extra_piece(seg_width, gcell_width):
  """Calculates the span and extra space when a segment is placed.

  Args:
    seg_width: Width of the segment.
    gcell_width: Width of the grid cell.

  Returns:
    The number of spanned gcells by a segment, and the amount of occupied
    space at the last gcell by this segment.
  """
  # Could get span immediately by division, but avoiding division to reduce
  # floating point errors. Not performance optimized.
  span = -1
  extra_space = -1.0
  i = 0
  while extra_space < 0.0:
    # span can only be an odd number since we are placing the segment such that
    # its center is at the center of a grid cell.
    span = i * 2 + 1
    extra_space = (span * gcell_width - seg_width) / 2.0
    i += 1
  # extra_piece is space occuiped at the last gcell
  #
  #                            extra_space
  #                   extra_piece    |
  #                            |     |
  #                            V     V
  #                          <----><-->
  #   |----======|==========|======----|
  #    <-------->
  #   gcell_width
  #        <---------------------->
  #                seg_width
  #
  #  When seg_width is smaller than gcell_width, extra_piece is the length
  #  from the left end of gcell and right end of placed segment.
  #    <------------>
  #   |---===========---|
  #
  extra_piece = gcell_width - extra_space
  return span, extra_piece


def get_waste_ratio(segment_widths, gcell_width):
  """Returns 'waste' ratio.

  Args:
    segment_widths: list of widths of segments.
    gcell_width: width of a single grid cell.

  Returns:
    Waste ratio:
    A segment is placed such that the center point is at the center of a
    grid cell. Every segment is placed without overlaps. Then the space that
    is not occupied by a segment is divided by the total spanned width by these
    segments.
  """
  # TODO(mustafay): This function just returns the waste ratio for the given
  # order of the segments. It may be a good idea to call this function with
  # many different orderings, and average the results. Finding the perfect
  # packed ordering may not be a good metric, since it may not reflect the
  # actual placement situation.
  tot_width = sum(segment_widths)
  assert tot_width > 0.0
  index = 0
  prev_extra_piece = 0.0
  for segw in segment_widths:
    spanned_cells_int, extra_piece = get_span_and_extra_piece(segw, gcell_width)
    can_fit_in_one = (extra_piece + prev_extra_piece) <= gcell_width
    index += spanned_cells_int
    if can_fit_in_one:
      index -= 1
    prev_extra_piece = extra_piece
  # index starts from 0, so adding 1 to get the actual span.
  span = (index + 1) * gcell_width
  waste_ratio = (span - tot_width) / span
  return waste_ratio


def get_hard_macros(plc, include_fixed_macros):
  hard_macros = []
  for m in plc.get_macro_indices():
    if plc.is_node_soft_macro(m):
      continue
    if plc.is_node_fixed(m) and not include_fixed_macros:
      continue
    hard_macros.append(m)
  return hard_macros


def get_macro_widths_heights(plc, add_size, include_fixed_macros):
  """Returns the widths and heights of movable hard macros.

  The returned widths and heights are adjusted with macro bloating config and
  add_size.

  Args:
    plc: A PlacementCost instance.
    add_size: Add to segment sizes to leave space between macros.
    include_fixed_macros: If set, include the macro sizes for fixed macros.

  Returns:
    A tuple of list of widths and heights for hard macros.
  """
  macro_widths = []
  macro_heights = []
  for m in get_hard_macros(plc, include_fixed_macros):
    w, h = plc.get_node_width_height(m)
    macro_x_bloat = plc.get_macro_bloat_width()
    macro_y_bloat = plc.get_macro_bloat_height()
    macro_widths.append(w + macro_x_bloat + add_size)
    macro_heights.append(h + macro_y_bloat + add_size)
  return macro_widths, macro_heights


def is_grid_size_acceptable(rows, cols, max_grid_size):
  """Checks the grid size in each dimension."""
  return rows <= max_grid_size and cols <= max_grid_size


def is_num_grid_cells_acceptable(num_grid_cells, min_num_grid_cells,
                                 max_num_grid_cells):
  """Checks the number of grid cells."""
  return (num_grid_cells <= max_num_grid_cells and
          num_grid_cells >= min_num_grid_cells)


def is_grid_cell_aspect_ratio_ok(gcell_width, gcell_height, max_aspect_ratio):
  """Checks the aspect ratio of a grid cell."""
  if gcell_width > gcell_height:
    aspect_ratio = gcell_width / gcell_height
  else:
    aspect_ratio = gcell_height / gcell_width
  return aspect_ratio <= max_aspect_ratio


def get_available_positions(mask):
  return [pos for pos, m in enumerate(mask) if m]


def get_empty_cells_ratio(plc):
  """Calculates the ratio of empty cells in the density matrix."""
  # If the density of a grid cell is greater than used_threshold, it's
  # not counted as empty.
  used_threshold = 1e-5
  is_empty = [d < used_threshold for d in plc.get_grid_cells_density()]
  return float(sum(is_empty)) / len(is_empty)


def try_placing(plc, hard_macros):
  """Places each macro in the list, one by one, at the first available location.

  Args:
    plc: placement_cost object
    hard_macros: node indices of movable hard macros to be placed.

  Returns:
    True if all macros were placed successfully, False otherwise.
  """
  # In case plc is loaded with fixed macros.
  for m in hard_macros:
    plc.unfix_node_coord(m)
  plc.unplace_all_nodes()
  for m in hard_macros:
    mask = plc.get_node_mask(m)
    avail_pos = get_available_positions(mask)
    if not avail_pos:
      return False
    plc.place_node(m, avail_pos[0])
  return True


def get_key_metric(data_value):
  """A metric to get an understanding of how good a col/row choice is."""
  return (data_value.empty_ratio + (1.0 - data_value.hor_waste) +
          (1.0 - data_value.ver_waste))


def get_grid_choices(plc, min_num, max_num, max_grid_size, min_num_grid_cells,
                     max_num_grid_cells, max_aspect_ratio, add_size,
                     include_fixed_macros):
  """Returns all possible grid number of columns/rows and their metrics.

  Args:
    plc: placement_cost object.
    min_num: minimum number of columns and rows during sweep.
    max_num: maximum number of columns and rows during sweep.
    max_grid_size:  maximum acceptable number of grid in each dimenssion.
    min_num_grid_cells: minimum acceptable number of grid cells.
    max_num_grid_cells: maximum acceptable number of grid cells.
    max_aspect_ratio: Maximum aspect ratio of a grid cell. (Make sure this
      number is gretaer than 1 since minimum aspect ratio is used as
      1/max_aspect_ratio.)
    add_size: bloat macro sizes by this amount to allow extra space between.
    include_fixed_macros: If set, include the fixed macros in the grid
      selection.

  Returns:
    A dict with key (num_cols, num_rows), and value is dataclass of
    key_metric, emty_ratio, horizontal waste ratio, vertical waste ratio,
    num grid cells.
    Empty_ratio is calculated by placing all macros in the canvas.
    Horizontal waste is just using widths of macros and grid cell width.
    Vertical waste is just using heights of macros and grid cell height.
  """
  # Select only hard macros for placement, and size selection process.
  hard_macros = get_hard_macros(plc, include_fixed_macros)
  if not hard_macros:
    print('No hard macros found in the design!')
    return None
  # grid_choices is a dict with key "num_cols num_rows" and value is list of
  # key_metric, emty_ratio, horizontal waste ratio, vertical waste ratio,
  # num grid cells.
  grid_choices = dict()

  @dataclasses.dataclass
  class ValueData:
    key_metric: float
    empty_ratio: float
    hor_waste: dict
    ver_waste: dict
    num_gcells: int

  # hor_waste and ver_waste are metrics for wasted space in either direction
  # using only widths of macros in horizontal and heights in vertical
  # directions.
  hor_waste = dict()
  ver_waste = dict()
  macro_widths, macro_heights = get_macro_widths_heights(
      plc, add_size, include_fixed_macros)
  canvas_width, canvas_height = plc.get_canvas_width_height()
  # Loop through all combinations of number of cols and rows.
  for rows in range(min_num, max_num):
    for cols in range(min_num, max_num):
      gcell_width = canvas_width / cols
      gcell_height = canvas_height / rows
      num_gcells = cols * rows
      if not is_grid_size_acceptable(rows, cols, max_grid_size):
        continue
      if not is_num_grid_cells_acceptable(num_gcells, min_num_grid_cells,
                                          max_num_grid_cells):
        continue
      if not is_grid_cell_aspect_ratio_ok(gcell_width, gcell_height,
                                          max_aspect_ratio):
        continue
      plc.set_placement_grid(cols, rows)
      # TODO(mustafay): placement do not take add_size into account, add a
      # resize utility to placement_cost interface.
      if not try_placing(plc, hard_macros):
        continue
      # Only calculate hor_waste, ver_waste if needed.
      if cols not in hor_waste:
        hor_waste[cols] = get_waste_ratio(macro_widths, gcell_width)
      if rows not in ver_waste:
        ver_waste[rows] = get_waste_ratio(macro_heights, gcell_height)
      d = ValueData(
          key_metric=0,
          empty_ratio=get_empty_cells_ratio(plc),
          hor_waste=hor_waste[cols],
          ver_waste=ver_waste[rows],
          num_gcells=cols * rows)
      # Fill in the key_metric based on the rest of the metrics.
      d.key_metric = get_key_metric(d)
      grid_choices[(cols, rows)] = d
  return grid_choices


def select_from_grid_choices(grid_choices, tolerance=0.1, print_best_n=0):
  """Chooses the best number of cols and rows based on the key metric."""
  assert grid_choices
  sorted_list = sorted(
      list(grid_choices.items()), key=lambda x: x[1].key_metric, reverse=True)

  # If the key metric is very close to the best (first item in the sorted list)
  # solution, choose the one with the smallest number of grid cells.
  # The key metric (emptiness) is maximized, so subtracting the tolerance
  # and checking the metric of the next items to be bigger than that.
  best_key_metric_tolerance = sorted_list[0][1].key_metric * (1.0 - tolerance)
  # The idea is to choose the minimum number of grid cells for which the
  # metric is not too far off.
  # TODO(mustafay): visualize and understand the impact of this choice.
  # Also, must understand the impact of number of grid cells on the RL/SA run
  # time vs QoR of the PnR tool.
  qualified = [(k, v)
               for (k, v) in sorted_list
               if v.key_metric >= best_key_metric_tolerance]
  # best choice is among the qualified results whose key_metric is within
  # a percentage of the highest key_metric. This is trying to reduce the
  # number of grid cells while keeping the packing metrics acceptable.
  best_choice = max(qualified, key=lambda x: x[1].key_metric / x[1].num_gcells)

  if print_best_n:
    print_best_n = max(min(print_best_n, len(grid_choices)), len(qualified))
    print('Printing first %d best choices and metrics' % print_best_n)
    for index, x in enumerate(sorted_list[0:print_best_n]):
      print('%s%d - %s' % ('-> ' if x[0] == best_choice[0] else '', index, x))

  return best_choice[0]


def place_only_macros(plc, random_order=False):
  """Utility function to place only hard macros to the first available loc.

  Args:
    plc: placement_cost object
    random_order: shuffle the order of hard macros to visualize if different
      ordering yields a different packing.

  Returns:
    True if all macros are placed, False is one fails to be placed.

  This is used to visualize packing of hard macros with current setting of
  canvas and grid sizes.
  """
  hard_macros = get_hard_macros(plc, FLAGS.grid_select_include_fixed_macros)
  if random_order:
    random.shuffle(hard_macros)
  # In case plc is loaded with fixed macros.
  for m in hard_macros:
    plc.unfix_node_coord(m)
  plc.unplace_all_nodes()
  for m in hard_macros:
    mask = plc.get_node_mask(m)
    avail_pos = get_available_positions(mask)
    if not avail_pos:
      return False
    plc.place_node(m, avail_pos[0])
  return True
