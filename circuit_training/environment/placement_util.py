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

import os
import datetime
import re
import textwrap
from typing import Dict, Iterator, List, Optional, Tuple, Any, Union

from absl import logging
from circuit_training.environment import plc_client
import numpy as np

# Internal gfile dependencies


def nodes_of_types(plc: plc_client.PlacementCost,
                   type_list: List[str]) -> Iterator[int]:
  """Yields the index of a node of certain types."""
  i = 0
  while True:
    node_type = plc.get_node_type(i)
    if not node_type:
      break
    if node_type in type_list:
      yield i
    i += 1


def get_node_xy_coordinates(
    plc: plc_client.PlacementCost) -> Dict[int, Tuple[float, float]]:
  """Returns all node x,y coordinates (canvas) in a dict."""
  node_coords = dict()
  for node_index in nodes_of_types(plc, ['MACRO', 'STDCELL', 'PORT']):
    if plc.is_node_placed(node_index):
      node_coords[node_index] = plc.get_node_location(node_index)
  return node_coords


def get_macro_orientations(plc: plc_client.PlacementCost) -> Dict[int, int]:
  """Returns all macros' orientations in a dict."""
  macro_orientations = dict()
  for node_index in nodes_of_types(plc, ['MACRO']):
    macro_orientations[node_index] = plc.get_macro_orientation(node_index)
  return macro_orientations


def restore_node_xy_coordinates(
    plc: plc_client.PlacementCost,
    node_coords: Dict[int, Tuple[float, float]]) -> None:
  for node_index, coords in node_coords.items():
    if not plc.is_node_fixed(node_index):
      plc.update_node_coords(node_index, coords[0], coords[1])


def restore_macro_orientations(plc: plc_client.PlacementCost,
                               macro_orientations: Dict[int, int]) -> None:
  for node_index, orientation in macro_orientations.items():
    plc.update_macro_orientation(node_index, orientation)


def extract_attribute_from_comments(attribute: str,
                                    filenames: List[str]) -> Optional[str]:
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
    filenames: Union[str, List[str]]) -> Optional[List[List[float]]]:
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
    filenames: List[str]) -> Optional[Tuple[float, float, int, int]]:
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


def fix_port_coordinates(plc: plc_client.PlacementCost) -> None:
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
    netlist_file: str,
    init_placement: Optional[str] = None,
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

  plc = plc_client.PlacementCost(netlist_file, macro_macro_x_spacing,
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


def get_node_type_counts(plc: plc_client.PlacementCost) -> Dict[str, int]:
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


def make_blockage_text(plc: plc_client.PlacementCost) -> str:
  ret = ''
  for blockage in plc.get_blockages():
    ret += 'Blockage : {}\n'.format(' '.join([str(b) for b in blockage]))
  return ret


def save_placement(plc: plc_client.PlacementCost,
                   filename: str,
                   user_comments: str = '') -> None:
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


def get_ordered_node_indices(mode: str,
                             plc: plc_client.PlacementCost,
                             exclude_fixed_nodes: bool = True) -> List[int]:
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


def extract_parameters_from_comments(
    filename: str) -> Tuple[float, float, int, int]:
  """Parses the file's comments section, tries to extract canvas/grid sizes.

  Args:
    filename: protobuf file or a plc file.

  Returns:
    Tuple of canvas_width, canvas_height, grid_cols, grid_rows
  """
  filename0 = filename.split(',')[0]
  canvas_width, canvas_height = None, None
  grid_cols, grid_rows = None, None
  with open(filename0, 'r') as infile:
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
  return canvas_width, canvas_height, grid_cols, grid_rows


def get_routing_resources() -> Dict[str, float]:
  """Currently we only use default parameter settings.

  In the future, for specific project, the resources may need to be tuned.

  Returns:
    Routing resources.
  """

  return {
      'horizontal_routes_per_micron': 57.031,
      'vertical_routes_per_micron': 56.818,
      'macro_horizontal_routing_allocation': 39.583,
      'macro_vertical_routing_allocation': 30.303,
  }


def nodes_of_types(plc: plc_client.PlacementCost, type_list: List[str]):
  """Yields the index of a node of certain types."""
  i = 0
  while True:
    node_type = plc.get_node_type(i)
    if not node_type:
      break
    if node_type in type_list:
      yield i
    i += 1


def num_nodes_of_type(plc, node_type):
  """Returns number of node of a particular type."""
  count = 0
  for _ in nodes_of_types(plc, [node_type]):
    count += 1
  return count


def extract_blockages_from_tcl(filename: str,
                               block_name: str,
                               canvas_width: float,
                               canvas_height: float,
                               is_rectilinear: bool = False):
  """Reads blockage information from a given tcl file."""
  # Assumptions: project is viperlite or viperfish.
  # This is not a TCL parser, it just reads in a line of the format:
  # dict set ::clockstrap <block name> <blockage index> <corner> <float number>
  # corner is expected to be one of lly, ury.
  blockage_info = dict()
  try:
    with open(filename, 'r') as infile:
      for line in infile:
        if line.startswith('dict set ::clockstrap '):
          block, index, corner, value = line.split()[3:7]
          if block != block_name:
            continue
          blockage_info[corner + index] = float(value)
  except gfile.FileError:
    logging.error('could not read file %s', filename)
    return []
  blockages = []

  if is_rectilinear:
    # Use blockage to model rectilinear floorplan.
    index = 0
    while ('llx' + str(index) in blockage_info and
           'lly' + str(index) in blockage_info and
           'urx' + str(index) in blockage_info and
           'ury' + str(index) in blockage_info):
      minx = blockage_info['llx' + str(index)]
      maxx = blockage_info['urx' + str(index)]
      miny = blockage_info['lly' + str(index)]
      maxy = blockage_info['ury' + str(index)]
      if minx < 0:
        raise ValueError(f'Illegal blockage at index {index}: llx {minx} < 0')
      if maxx > canvas_width:
        raise ValueError(
            f'Illegal blockage at index {index}: urx {maxx} > canvas '
            f'width {canvas_width}')
      if miny < 0:
        raise ValueError(f'Illegal blockage at index {index}: lly {miny} < 0')
      if maxy > canvas_height:
        raise ValueError(
            f'Illegal blockage at index {index}: ury {maxy} > canvas '
            f'height {canvas_height}')
      blockages.append([minx, miny, maxx, maxy, 1])
      index += 1
  else:
    # Fully horizontal or vertical blockage.
    # Horizontal straps.
    index = 0
    while 'lly' + str(index) in blockage_info and 'ury' + str(
        index) in blockage_info:
      minx = 0.0
      maxx = canvas_width
      miny = blockage_info['lly' + str(index)]
      maxy = blockage_info['ury' + str(index)]
      blockages.append([minx, miny, maxx, maxy, 1])
      index += 1
    # We don't have any vertical straps, now. Should we still support it?
    # Vertical straps.
    index = 0
    while 'llx' + str(index) in blockage_info and 'urx' + str(
        index) in blockage_info:
      minx = blockage_info['llx' + str(index)]
      maxx = blockage_info['urx' + str(index)]
      miny = 0.0
      maxy = canvas_height
      blockages.append([minx, miny, maxx, maxy, 1])
      index += 1
  return blockages


def get_ascii_picture(vect: List[Any],
                      cols: int,
                      rows: int,
                      scale: float = 10) -> str:
  """Returns an ascii picture for the input as a human readable matrix."""
  ret_str = '   '
  for c in range(cols):
    ret_str += '|' + str(int(c / 10) % 10)
  ret_str += '|\n   '
  for c in range(cols):
    ret_str += '|' + str(c % 10)
  ret_str += '|\n   -' + '-' * 2 * cols + '\n'
  for r in range(rows - 1, -1, -1):
    ret_str += format('%3d' % r)
    for c in range(cols):
      mindex = r * cols + c
      val = int(scale * vect[mindex])
      if val > scale:
        ret_str += '|!'
      elif val == scale:
        ret_str += '|#'
      elif val == 0:
        ret_str += '| '
      else:
        ret_str += '|' + str(val)
    ret_str += '|\n'
  ret_str += '   -' + '-' * 2 * cols + '\n'
  return ret_str


def get_hard_macro_density_map(plc: plc_client.PlacementCost) -> List[float]:
  """Returns the placement density map for hard macros only."""
  # Unplaces all standard cells and soft macros, so that grid cell density
  # only contains hard macros.
  placements_to_restore = dict()
  for node_index in nodes_of_types(plc, ['STDCELL']):
    if plc.is_node_placed(node_index):
      placements_to_restore[node_index] = plc.get_node_location(node_index)
      plc.unplace_node(node_index)
  for node_index in nodes_of_types(plc, ['MACRO']):
    if plc.is_node_soft_macro(node_index) and plc.is_node_placed(node_index):
      placements_to_restore[node_index] = plc.get_node_location(node_index)
      plc.unplace_node(node_index)
  hard_macro_density = plc.get_grid_cells_density()
  check_boundary = plc.get_canvas_boundary_check()
  # Restores placements, but original placement may be illegal (outside canvas
  # area), ignore those cases.
  plc.set_canvas_boundary_check(False)
  for node_index, coords in placements_to_restore.items():
    plc.update_node_coords(node_index, coords[0], coords[1])
  plc.set_canvas_boundary_check(check_boundary)
  return hard_macro_density


def save_placement_with_info(plc: plc_client.PlacementCost,
                             filename: str,
                             user_comments: str = '') -> None:
  """Saves the placement file with some information in the comments section."""
  cols, rows = plc.get_grid_num_columns_rows()
  width, height = plc.get_canvas_width_height()
  hor_routes, ver_routes = plc.get_routes_per_micron()
  hor_macro_alloc, ver_macro_alloc = plc.get_macro_routing_allocation()
  smooth = plc.get_congestion_smooth_range()
  init_placement_config = ''
  # Do not change the format of the comments section before updating
  # extract_parameters_from_comments and extract_netlist_file_from_comments
  # functions.
  info = textwrap.dedent("""\
    Placement file for Circuit Training
    Source input file(s) : {src_filename}
    This file : {filename}
    Original initial placement : {init_placement_config}
    Date : {date}
    Columns : {cols}  Rows : {rows}
    Width : {width:.3f}  Height : {height:.3f}
    Area (stdcell+macros) : {area}
    Wirelength : {wl:.3f}
    Wirelength cost : {wlc:.4f}
    Congestion cost : {cong:.4f}
    Density cost : {density:.4f}
    Fake net cost : {fake_net:.4f}
    90% Congestion metric: {cong90}
    Project : {project}
    Block : {block_name}
    Routes per micron, hor : {hor_routes:.3f}  ver : {ver_routes:.3f}
    Routes used by macros, hor : {hor_macro_alloc:.3f}  ver : {ver_macro_alloc:.3f}
    Smoothing factor : {smooth}
    Use incremental cost : {incr_cost}

    To view this file (most options are default):
    viewer_binary\
    --netlist_file {src_filename}\
    --canvas_width {width} --canvas_height {height}\
    --grid_cols {cols} --grid_rows={rows}\
    --init_placement {filename}\
    --project {project}\
    --block_name {block_name}\
    --congestion_smooth_range {smooth}\
    --overlap_threshold {overlap_threshold}\
    --noboundary_check
    or you can simply run:
    viewer_binary\
    --init_placement {filename}
  """.format(
      src_filename=plc.get_source_filename(),
      filename=filename,
      init_placement_config=init_placement_config,
      date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      cols=cols,
      rows=rows,
      width=width,
      height=height,
      area=plc.get_area(),
      wl=plc.get_wirelength(),
      wlc=plc.get_cost(),
      cong=plc.get_congestion_cost(),
      cong90=plc.get_congestion_cost_threshold(0.9),
      density=plc.get_density_cost(),
      fake_net=plc.get_fake_net_cost(),
      project=plc.get_project_name(),
      block_name=plc.get_block_name(),
      hor_routes=hor_routes,
      ver_routes=ver_routes,
      hor_macro_alloc=hor_macro_alloc,
      ver_macro_alloc=ver_macro_alloc,
      smooth=smooth,
      incr_cost=plc.get_use_incremental_cost(),
      overlap_threshold=plc.get_overlap_threshold()))

  info += '\n' + make_blockage_text(plc) + '\n'
  info += '\nCounts of node types:\n'
  node_type_counts = get_node_type_counts(plc)
  for node_type in sorted(node_type_counts):
    info += '{:<15} : {:>9}\n'.format(node_type + 's',
                                      node_type_counts[node_type])
  info += '\nHard Macro Placements:\n'
  info += get_ascii_picture(get_hard_macro_density_map(plc), cols, rows)
  info += '\nOverall Placement Density:\n'
  info += get_ascii_picture(plc.get_grid_cells_density(), cols, rows)
  info += '\nHorizontal Routing Congestion:\n'
  info += get_ascii_picture(plc.get_horizontal_routing_congestion(), cols, rows)
  info += '\nVertical Routing Congestion:\n'
  info += get_ascii_picture(plc.get_vertical_routing_congestion(), cols, rows)
  if user_comments:
    info += '\nUser comments:\n' + user_comments + '\n'
  info += '\nnode_index x y orientation fixed'
  return plc.save_placement(filename, info)


def create_placement_cost_using_common_arguments(
    netlist_file: str,
    init_placement: Optional[str] = None,
    canvas_width: Optional[float] = None,
    canvas_height: Optional[float] = None,
    grid_cols: Optional[int] = None,
    grid_rows: Optional[int] = None,
    project: Optional[str] = None,
    block_name: Optional[str] = None,
    congestion_smooth_range: Optional[int] = None,
    overlap_threshold: Optional[float] = None,
    use_incremental_cost: Optional[bool] = None,
    boundary_check: Optional[bool] = None,
    blockages: Optional[List[List[float]]] = None,
    fix_ports: Optional[bool] = True) -> plc_client.PlacementCost:
  """Creates a placement_cost object using the common arguments."""
  if not project:
    logging.info('Reading project name from file.')
    project = extract_attribute_from_comments('Project',
                                              [init_placement, netlist_file])
  if init_placement and not block_name:
    logging.info('Reading block name from file.')
    block_name = extract_attribute_from_comments('Block',
                                                 [init_placement, netlist_file])
  if not block_name:
    logging.warning('block_name is not set. Please add the block_name in:\n%s',
                    init_placement)

  plc = plc_client.PlacementCost(netlist_file)
  # Create blockages.
  if blockages is None:
    # Try to read blockages from input files. To avoid file I/O, pass blockages,
    # or an empty list if there are none.
    logging.info('Reading blockages from file.')
    for filename in [netlist_file, init_placement]:
      if filename is None:
        continue
      blockages = get_blockages_from_comments(filename)
      # Only read blockages from one file.
      if blockages:
        break
  if blockages:
    for blockage in blockages:
      plc.create_blockage(*blockage)
  # Give precedence to command line parameters for canvas/grid sizes.
  canvas_size_set = False
  if canvas_width and canvas_height:
    plc.set_canvas_size(canvas_width, canvas_height)
    canvas_size_set = True
  grid_size_set = False
  if grid_cols and grid_rows:
    grid_size_set = True
    plc.set_placement_grid(grid_cols, grid_rows)
  # Extract and set canvas, grid sizes if they are not already set.
  if not canvas_size_set or not grid_size_set:
    logging.info('Reading netlist sizes from file.')
    for filename in [netlist_file, init_placement]:
      if filename is None:
        continue
      sizes = extract_parameters_from_comments(filename)
      canvas_width, canvas_height, grid_cols, grid_rows = sizes
      if canvas_width and canvas_height and not canvas_size_set:
        plc.set_canvas_size(canvas_width, canvas_height)
      if grid_cols and grid_rows and not grid_size_set:
        plc.set_placement_grid(grid_cols, grid_rows)

  routing_resources = get_routing_resources()
  plc.set_project_name(project or 'unset_project')
  plc.set_block_name(block_name or 'unset_block')
  plc.set_routes_per_micron(routing_resources['horizontal_routes_per_micron'],
                            routing_resources['vertical_routes_per_micron'])
  plc.set_macro_routing_allocation(
      routing_resources['macro_horizontal_routing_allocation'],
      routing_resources['macro_vertical_routing_allocation'])
  plc.set_congestion_smooth_range(congestion_smooth_range)
  plc.set_overlap_threshold(overlap_threshold)
  plc.set_canvas_boundary_check(boundary_check)

  # Set macros to initial locations.
  if init_placement:
    logging.info('Reading init_placement from file %s', init_placement)
    # I/O is forbidden in forked child processes.
    # Reads init placement from file only if init_locations are not provided.
    plc.restore_placement(init_placement)

  if fix_ports:
    fix_port_coordinates(plc)
  plc.set_use_incremental_cost(use_incremental_cost)

  return plc


def get_node_locations(plc: plc_client.PlacementCost) -> Dict[int, int]:
  """Returns all node grid locations (macros and stdcells) in a dict."""
  node_locations = dict()
  for i in nodes_of_types(plc, ['MACRO', 'STDCELL']):
    node_locations[i] = plc.get_grid_cell_of_node(i)
  return node_locations


def get_node_ordering_by_size(plc: plc_client.PlacementCost) -> List[int]:
  """Returns the list of nodes (macros and stdcells) ordered by area."""
  node_areas = dict()
  for i in nodes_of_types(plc, ['MACRO', 'STDCELL']):
    if plc.is_node_fixed(i):
      continue
    w, h = plc.get_node_width_height(i)
    node_areas[i] = w * h
  return sorted(node_areas, key=node_areas.get, reverse=True)


def grid_locations_near(plc: plc_client.PlacementCost,
                        start_grid_index: int) -> Iterator[int]:
  """Yields node indices closest to the start_grid_index."""
  # Starting from the start_grid_index, it goes around the area from closest
  # (manhattan distance) to the farthest. For example, if the start grid index
  # is at 0, the order of the next grid cells will be like:
  #          24
  #       22 12 23
  #    20 10  4 11 21
  # 18  8  2  0  3  9 19
  #    16  6  1  7 17
  #       14  5 15
  #          13
  cols, rows = plc.get_grid_num_columns_rows()
  start_col, start_row = start_grid_index % cols, int(start_grid_index / cols)
  # TODO(mustafay): This may be improved, but it's not crucial now.
  for distance in range(cols + rows):
    for row_offset in range(-distance, distance + 1):
      for col_offset in range(-distance, distance + 1):
        if abs(row_offset) + abs(col_offset) != distance:
          continue
        new_col = start_col + col_offset
        new_row = start_row + row_offset
        if new_col < 0 or new_row < 0 or new_col >= cols or new_row >= rows:
          continue
        yield int(new_col + new_row * cols)


def place_near(plc: plc_client.PlacementCost, node_index: int,
               location: int) -> bool:
  """Places a node (legally) closest to the given location.

  Args:
    plc: placement_cost object.
    node_index: index of a node.
    location: target grid cell location. (row * num_cols + num_cols)

  Returns:
    True on success, False if this node was not placed on any grid legally.
  """
  for loc in grid_locations_near(plc, location):
    if plc.can_place_node(node_index, loc):
      plc.place_node(node_index, loc)
      return True
  return False


def disconnect_high_fanout_nets(plc: plc_client.PlacementCost,
                                max_allowed_fanouts: int = 500) -> None:
  high_fanout_nets = []
  for i in nodes_of_types(plc, ['PORT', 'STDCELL', 'MACRO_PIN']):
    num_fanouts = len(plc.get_fan_outs_of_node(i))
    if num_fanouts > max_allowed_fanouts:
      print('Disconnecting node: {} with {} fanouts.'.format(
          plc.get_node_name(i), num_fanouts))
      high_fanout_nets.append(i)
  plc.disconnect_nets(high_fanout_nets)


def legalize_placement(plc: plc_client.PlacementCost) -> bool:
  """Places the nodes to legal positions snapping to grid cells."""
  # Unplace all except i/o's.
  fix_port_coordinates(plc)
  # First save each node's locations on the grid.
  # Note that the orientations are not changed by this utility, we do not
  # need saving/restoring existing orientations.
  node_locations = get_node_locations(plc)
  previous_xy_coords = get_node_xy_coordinates(plc)
  total_macro_displacement = 0
  total_macros = 0
  plc.unplace_all_nodes()
  # Starting with the biggest, place them trying to be as close as possible
  # to the original position.
  ordered_nodes = get_node_ordering_by_size(plc)
  for node in ordered_nodes:
    if not place_near(plc, node, node_locations[node]):
      print('Could not place node')
      return False
    if node in previous_xy_coords and not plc.is_node_soft_macro(node):
      x, y = plc.get_node_location(node)
      px, py = previous_xy_coords[node]
      print('x/y displacement: dx = {}, dy = {}, macro: {}'.format(
          x - px, y - py, plc.get_node_name(node)))
      total_macro_displacement += abs(x - px) + abs(y - py)
      total_macros += 1
  print('Total macro displacement: {}, avg: {}'.format(
      total_macro_displacement, total_macro_displacement / total_macros))
  return True
