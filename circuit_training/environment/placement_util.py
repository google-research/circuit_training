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
"""A collection of non-prod utility functions for placement.

All the dependencies in this files should be non-prod.
"""

import datetime
import re
import textwrap
from typing import Dict, Iterator, List, Optional, Text, Tuple

from absl import logging
from circuit_training.environment import plc_client
import numpy as np

# Internal gfile dependencies


def nodes_of_types(plc: plc_client.PlacementCost,
                   type_list: List[Text]) -> Iterator[int]:
  """Yields the index of a node of certain types."""
  i = 0
  while True:
    node_type = plc.get_node_type(i)
    if not node_type:
      break
    if node_type in type_list:
      yield i
    i += 1


def extract_attribute_from_comments(attribute: Text,
                                    filenames: List[Text]) -> Optional[Text]:
  """Parses the files' comments section, tries to extract the attribute.

  Args:
    attribute: attribute to look for (case sensetive).
    filenames: List of protobuf file or a plc file.

  Returns:
    Attribute name string, or None if not found.
  """
  for filename in filenames:
    if filename:
      f = filename.split(',')[0]
      if f:
        with open(f, 'rt') as infile:
          for line in infile:
            if line.startswith('#'):
              match = re.search(fr'{attribute} : ([-\w]+)', line)
              if match:
                return match.group(1)
            else:
              # Do not parse the rest of the file, since all the comments are at
              # the top.
              break
  return None


def get_blockages_from_comments(
    filenames: List[Text]) -> Optional[List[List[float]]]:
  """Returns list of blockages if they exist in the file's comments section."""
  for filename in filenames:
    if not filename:
      continue
    blockages = []
    # Read the first file if filename is comma separated list.
    # Expected blockage info line format is:
    # "# Blockage : <float> <float> <float> <float> <float>"
    # where first four float numbers correspond to minx, miny, maxx, maxy of
    # the rectangular region, the fifth one is the blockage rate. It's usually
    # set to 1.
    try:
      with open(filename, 'rt') as infile:
        for line in infile:
          if line.startswith('# Blockage : '):
            blockages.append([float(x) for x in line.split()[3:8]])
          elif not line.startswith('#'):
            break
    except OSError:
      logging.error('could not read file %s.', filename)
    if blockages:
      return blockages


def extract_sizes_from_comments(
    filenames: List[Text]) -> Optional[Tuple[float, float, int, int]]:
  """Parses the file's comments section, tries to extract canvas/grid sizes.

  Args:
    filenames: A list of netlist (.pb.txt) or placement (.plc) files.

  Returns:
    Tuple of canvas_width, canvas_height, grid_cols, grid_rows
  """
  for filename in filenames:
    if not filename:
      continue
    canvas_width, canvas_height = None, None
    grid_cols, grid_rows = None, None
    with open(filename, 'rt') as infile:
      for line in infile:
        if line.startswith('#'):
          fp_re = re.search(
              r'FP bbox: \{([\d\.]+) ([\d\.]+)\} \{([\d\.]+) ([\d\.]+)\}', line)
          if fp_re:
            canvas_width = float(fp_re.group(3))
            canvas_height = float(fp_re.group(4))
            continue
          plc_wh = re.search(r'Width : ([\d\.]+)  Height : ([\d\.]+)', line)
          if plc_wh:
            canvas_width = float(plc_wh.group(1))
            canvas_height = float(plc_wh.group(2))
            continue
          plc_cr = re.search(r'Columns : ([\d]+)  Rows : ([\d]+)', line)
          if plc_cr:
            grid_cols = int(plc_cr.group(1))
            grid_rows = int(plc_cr.group(2))
        else:
          # Do not parse the rest of the file, since all the comments are at the
          # top.
          break
    if canvas_width and canvas_height and grid_cols and grid_rows:
      return canvas_width, canvas_height, grid_cols, grid_rows


def fix_port_coordinates(plc: plc_client.PlacementCost):
  """Find all ports and fix their coordinates.

  Args:
    plc: the placement cost object.
  """
  for node in nodes_of_types(plc, ['PORT']):
    plc.fix_node_coord(node)


# The routing capacities are calculated based on the public information about
# 7nm technology (https://en.wikichip.org/wiki/7_nm_lithography_process)
# with an arbitary, yet reasonable, assumption of 18% of the tracks for
# the power grids.
def create_placement_cost(
    netlist_file: Text,
    init_placement: Optional[Text] = None,
    overlap_threshold: float = 4e-3,
    congestion_smooth_range: int = 2,
    # TODO(b/211039937): Increase macro spacing to 3-5um, after matching the
    # performance for Ariane.
    macro_macro_x_spacing: float = 0.1,
    macro_macro_y_spacing: float = 0.1,
    boundary_check: bool = False,
    horizontal_routes_per_micron: float = 70.33,
    vertical_routes_per_micron: float = 74.51,
    macro_horizontal_routing_allocation: float = 51.79,
    macro_vertical_routing_allocation: float = 51.79,
) -> plc_client.PlacementCost:
  """Creates a placement_cost object.

  Args:
    netlist_file: Path to the netlist proto text file.
    init_placement: Path to the inital placement .plc file.
    overlap_threshold: Used for macro overlap detection.
    congestion_smooth_range: Smoothing factor used for congestion estimation.
      Congestion is distributed to this many neighboring columns/rows.'
    macro_macro_x_spacing: Macro-to-macro x spacing in microns.
    macro_macro_y_spacing: Macro-to-macro y spacing in microns.
    boundary_check: Do a boundary check during node placement.
    horizontal_routes_per_micron: Horizontal route capacity per micros.
    vertical_routes_per_micron: Vertical route capacity per micros.
    macro_horizontal_routing_allocation: Macro horizontal routing allocation.
    macro_vertical_routing_allocation: Macro vertical routing allocation.

  Returns:
    A PlacementCost object.
  """
  if not netlist_file:
    raise ValueError('netlist_file should be provided.')

  block_name = extract_attribute_from_comments('Block',
                                               [init_placement, netlist_file])
  if not block_name:
    logging.warning(
        'block_name is not set. '
        'Please add the block_name in:\n%s\nor in:\n%s', netlist_file,
        init_placement)

  plc = plc_client.PlacementCost(
      netlist_file,
      macro_macro_x_spacing,
      macro_macro_y_spacing)

  blockages = get_blockages_from_comments([netlist_file, init_placement])
  if blockages:
    for blockage in blockages:
      plc.create_blockage(*blockage)

  sizes = extract_sizes_from_comments([netlist_file, init_placement])
  if sizes:
    canvas_width, canvas_height, grid_cols, grid_rows = sizes
    if canvas_width and canvas_height and grid_cols and grid_rows:
      plc.set_canvas_size(canvas_width, canvas_height)
      plc.set_placement_grid(grid_cols, grid_rows)

  plc.set_project_name('circuit_training')
  plc.set_block_name(block_name or 'unset_block')
  plc.set_routes_per_micron(horizontal_routes_per_micron,
                            vertical_routes_per_micron)
  plc.set_macro_routing_allocation(macro_horizontal_routing_allocation,
                                   macro_vertical_routing_allocation)
  plc.set_congestion_smooth_range(congestion_smooth_range)
  plc.set_overlap_threshold(overlap_threshold)
  plc.set_canvas_boundary_check(boundary_check)
  plc.make_soft_macros_square()
  if init_placement:
    plc.restore_placement(init_placement)
    fix_port_coordinates(plc)

  return plc


def get_node_type_counts(plc: plc_client.PlacementCost) -> Dict[Text, int]:
  """Returns number of each type of nodes in the netlist.

  Args:
    plc: the placement cost object.

  Returns:
    Number of each type of node in a dict.
  """
  counts = {
      'MACRO': 0,
      'STDCELL': 0,
      'PORT': 0,
      'MACRO_PIN': 0,
      'SOFT_MACRO': 0,
      'HARD_MACRO': 0,
      'SOFT_MACRO_PIN': 0,
      'HARD_MACRO_PIN': 0
  }

  for node_index in nodes_of_types(plc,
                                   ['MACRO', 'STDCELL', 'PORT', 'MACRO_PIN']):
    node_type = plc.get_node_type(node_index)
    counts[node_type] += 1
    if node_type == 'MACRO':
      if plc.is_node_soft_macro(node_index):
        counts['SOFT_MACRO'] += 1
      else:
        counts['HARD_MACRO'] += 1
    if node_type == 'MACRO_PIN':
      ref_id = plc.get_ref_node_id(node_index)
      if plc.is_node_soft_macro(ref_id):
        counts['SOFT_MACRO_PIN'] += 1
      else:
        counts['HARD_MACRO_PIN'] += 1
  return counts


def make_blockage_text(plc: plc_client.PlacementCost) -> Text:
  ret = ''
  for blockage in plc.get_blockages():
    ret += 'Blockage : {}\n'.format(' '.join([str(b) for b in blockage]))
  return ret


def save_placement(plc: plc_client.PlacementCost,
                   filename: Text,
                   user_comments: Text = '') -> None:
  """Saves the placement file with some information in the comments section."""
  cols, rows = plc.get_grid_num_columns_rows()
  width, height = plc.get_canvas_width_height()
  hor_routes, ver_routes = plc.get_routes_per_micron()
  hor_macro_alloc, ver_macro_alloc = plc.get_macro_routing_allocation()
  smooth = plc.get_congestion_smooth_range()
  info = textwrap.dedent("""\
    Placement file for Circuit Training
    Source input file(s) : {src_filename}
    This file : {filename}
    Date : {date}
    Columns : {cols}  Rows : {rows}
    Width : {width:.3f}  Height : {height:.3f}
    Area : {area}
    Wirelength : {wl:.3f}
    Wirelength cost : {wlc:.4f}
    Congestion cost : {cong:.4f}
    Density cost : {density:.4f}
    Project : {project}
    Block : {block_name}
    Routes per micron, hor : {hor_routes:.3f}  ver : {ver_routes:.3f}
    Routes used by macros, hor : {hor_macro_alloc:.3f}  ver : {ver_macro_alloc:.3f}
    Smoothing factor : {smooth}
    Overlap threshold : {overlap_threshold}
  """.format(
      src_filename=plc.get_source_filename(),
      filename=filename,
      date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      cols=cols,
      rows=rows,
      width=width,
      height=height,
      area=plc.get_area(),
      wl=plc.get_wirelength(),
      wlc=plc.get_cost(),
      cong=plc.get_congestion_cost(),
      density=plc.get_density_cost(),
      project=plc.get_project_name(),
      block_name=plc.get_block_name(),
      hor_routes=hor_routes,
      ver_routes=ver_routes,
      hor_macro_alloc=hor_macro_alloc,
      ver_macro_alloc=ver_macro_alloc,
      smooth=smooth,
      overlap_threshold=plc.get_overlap_threshold()))

  info += '\n' + make_blockage_text(plc) + '\n'
  info += '\nCounts of node types:\n'
  node_type_counts = get_node_type_counts(plc)
  for node_type in sorted(node_type_counts):
    info += '{:<15} : {:>9}\n'.format(node_type + 's',
                                      node_type_counts[node_type])
  if user_comments:
    info += '\nUser comments:\n' + user_comments + '\n'
  info += '\nnode_index x y orientation fixed'
  return plc.save_placement(filename, info)


def fd_placement_schedule(plc: plc_client.PlacementCost,
                          num_steps: Tuple[int, ...] = (100, 100, 100),
                          io_factor: float = 1.0,
                          move_distance_factors: Tuple[float,
                                                       ...] = (1.0, 1.0, 1.0),
                          attract_factor: Tuple[float,
                                                ...] = (100.0, 1.0e-3, 1.0e-5),
                          repel_factor: Tuple[float, ...] = (0.0, 1.0e6, 1.0e7),
                          use_current_loc: bool = False,
                          move_macros: bool = False) -> None:
  """A placement schedule that uses force directed method.

  Args:
    plc: The plc object.
    num_steps: Number of steps of the force-directed algorithm during each call.
    io_factor: I/O attract factor.
    move_distance_factors: Maximum distance relative to canvas size that a node
      can move in a single step of the force-directed algorithm.
    attract_factor: The spring constants between two connected nodes in the
      force-directed algorithm. The FD algorithm will be called size of this
      list times. Make sure that the size of fd_repel_factor has the same size.
    repel_factor: The repellent factor for spreading the nodes to avoid
      congestion in the force-directed algorithm.'
    use_current_loc: If true, use the current location as the initial location.
    move_macros: If true, also move the macros.
  """
  assert len(num_steps) == len(move_distance_factors)
  assert len(num_steps) == len(repel_factor)
  assert len(num_steps) == len(attract_factor)
  canvas_size = max(plc.get_canvas_width_height())
  max_move_distance = [
      f * canvas_size / s for s, f in zip(num_steps, move_distance_factors)
  ]
  move_stdcells = True
  log_scale_conns = False
  use_sizes = False
  plc.optimize_stdcells(use_current_loc, move_stdcells, move_macros,
                        log_scale_conns, use_sizes, io_factor, num_steps,
                        max_move_distance, attract_factor, repel_factor)


def get_ordered_node_indices(mode, plc, exclude_fixed_nodes=True):
  """Returns an ordering of node indices according to the specified mode.

  Args:
    mode: node ordering mode
    plc: placement cost object
    exclude_fixed_nodes: Whether fixed nodes should be excluded.

  Returns:
    Node indices sorted according to the mode.
  """
  macro_indices = plc.get_macro_indices()
  hard_macro_indices = [
      m for m in macro_indices if not plc.is_node_soft_macro(m)
  ]
  soft_macro_indices = [m for m in macro_indices if plc.is_node_soft_macro(m)]

  def macro_area(idx):
    w, h = plc.get_node_width_height(idx)
    return w * h

  if mode == 'descending_size_macro_first':
    ordered_indices = (
        sorted(hard_macro_indices, key=macro_area)[::-1] +
        sorted(soft_macro_indices, key=macro_area)[::-1])
  elif mode == 'random':
    np.random.shuffle(macro_indices)
    ordered_indices = macro_indices
  elif mode == 'random_macro_first':
    np.random.shuffle(hard_macro_indices)
    ordered_indices = hard_macro_indices + soft_macro_indices
  else:
    raise ValueError('{} is an unsupported node placement mode.'.format(mode))

  if exclude_fixed_nodes:
    ordered_indices = [m for m in ordered_indices if not plc.is_node_fixed(m)]
  return ordered_indices
