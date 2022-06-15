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
"""Grouping (clustering) of standard cells in circuit training netlist."""

import dataclasses
import math
import os
from typing import Any, Callable, Dict, Optional, Tuple

from absl import flags
from absl import logging

import sortedcontainers

from circuit_training.environment import placement_util
from circuit_training.environment import plc_client
from circuit_training.grouping import grid_size_selection
from circuit_training.grouping import grouping
from circuit_training.grouping import meta_netlist_convertor
from circuit_training.grouping import meta_netlist_util
from circuit_training.grouping import hmetis_util
# Internal gfile dependencies

flags.DEFINE_integer(
    'fixed_logic_levels', 1, 'Number of levels of logic '
    'to traverse during fixed groups assignment.')
flags.DEFINE_bool(
    'auto_grid_size', True, 'Choose num columns and rows automatically. '
    'See grid_size_selection flags.')
flags.DEFINE_bool('breakup', True, 'Break up groups after hmetis run')
flags.DEFINE_bool(
    'extract_blockages', True,
    'Extract blockage information from project specific TCL file')

flags.DEFINE_float(
    'cell_area_utilization', 0.5,
    'This is used to bloat soft macro areas. If it\'s 0.5, it means 50% '
    'utilization is targeted.')
flags.DEFINE_string(
    'blockage_file', None,
    'Floorplan blockage tcl file which specifies clock strap or macro blockage '
    'to model rectilinear floorplan.')
flags.DEFINE_bool(
    'is_rectilinear', False,
    'If True, the blockage spec will be interpreted for rectilinear floorplan.')
flags.DEFINE_integer('num_groups', 500,
                     'Number of std-cell groups (soft macros)')

FLAGS = flags.FLAGS


def setup_fixed_groups(grp: grouping.Grouping, output_dir: str,
                       fixed_logic_levels: int) -> str:
  fix_file = os.path.join(output_dir, 'metis_input.fix')
  if gfile.Exists(fix_file):
    update_groups_using_metis_output(grp, fix_file)
    return fix_file

  grp.setup_fixed_groups(fixed_logic_levels)
  grp.write_metis_fix_file(fix_file)
  return fix_file


def partition_netlist(plc: plc_client.PlacementCost, num_groups: int,
                      fixed_logic_levels: int, cell_area_utilization: float,
                      breakup: bool, output_dir: str, hmetis_opts: Any,
                      netlist_file: str) -> Optional[Tuple[str, str]]:
  """Partitions standard cells into groups by calling hmetis."""
  if not gfile.IsDirectory(output_dir):
    gfile.MakeDirs(output_dir)
  metis_file = os.path.join(output_dir, 'metis_input')

  logging.info('Writing metis compatible file: %s', metis_file)

  meta_netlist = meta_netlist_convertor.read_netlist(netlist_file)
  meta_netlist_util.set_canvas_columns_rows(meta_netlist,
                                            *plc.get_grid_num_columns_rows())
  meta_netlist_util.set_canvas_width_height(meta_netlist,
                                            *plc.get_canvas_width_height())
  meta_netlist_util.disconnect_high_fanout_nets(meta_netlist)

  grp = grouping.Grouping(meta_netlist)
  grp.set_cell_area_utilization(cell_area_utilization)
  fix_file = setup_fixed_groups(grp, output_dir, fixed_logic_levels)
  num_fixed_groups = grp.num_groups()
  if num_fixed_groups:
    logging.info('Adding %d (number of fixed groups) to number of metis groups',
                 num_fixed_groups)
    num_groups += num_fixed_groups
  else:
    fix_file = None
  if gfile.Exists(metis_file):
    logging.warning('Metis input file exists, skipping generation.')
  else:
    try:
      grp.write_metis_file(metis_file)
    except Exception as e:
      logging.exception('Can not write out metis file: %s', e)
      return None

  metis_out_file = hmetis_util.call_hmetis(
      graph_file=metis_file,
      fix_file=fix_file,
      n_parts=num_groups,
      ub_factor=hmetis_opts.ub_factor,
      n_runs=hmetis_opts.n_runs,
      c_type=hmetis_opts.c_type,
      r_type=hmetis_opts.r_type,
      v_cycle=hmetis_opts.v_cycle,
      reconst=hmetis_opts.reconst)
  if not metis_out_file:
    return None
  new_netlist, plc_file = write_new_netlist(plc, fix_file,
                                            cell_area_utilization, breakup,
                                            metis_out_file, output_dir,
                                            netlist_file)
  return new_netlist, plc_file


def get_break_up_threshold(plc: plc_client.PlacementCost) -> float:
  # Area of the canvas
  w, h = plc.get_canvas_width_height()
  area = w * h
  # The cut threshold is a function of the area of canvas.
  # The divisr for area is set to 16, meaning that the groups covering more
  # than 1/4th of the canvas (treated as square) in each dimension will be cut.
  return math.sqrt(area / 16.0)


def write_new_netlist(plc: plc_client.PlacementCost, fix_file: str,
                      cell_area_utilization: float, breakup: bool,
                      metis_output: str, output_dir: str,
                      netlist_file: str) -> Tuple[str, str]:
  """Writes out a partitioned netlist with soft macros.

  Args:
    plc: plc_client object
    fix_file: metis fix file
    cell_area_utilization: Used to bloat soft macro area.
    breakup: break up clusters that span a wide area
    metis_output: metis output file with group information
    output_dir: placing new files into this directory.
    netlist_file: Path to the original (non-clustred) netlist file.

  Returns:
    Tuple of name of the new netlist file and initial placement file.
  """
  filename = os.path.join(output_dir, 'netlist.pb.txt')

  meta_netlist = meta_netlist_convertor.read_netlist(netlist_file)
  meta_netlist_util.set_canvas_columns_rows(meta_netlist,
                                            *plc.get_grid_num_columns_rows())
  meta_netlist_util.set_canvas_width_height(meta_netlist,
                                            *plc.get_canvas_width_height())
  meta_netlist_util.disconnect_high_fanout_nets(meta_netlist)

  grp = grouping.Grouping(meta_netlist)
  grp.set_cell_area_utilization(cell_area_utilization)
  if fix_file:
    update_groups_using_metis_output(grp, fix_file)
  update_groups_using_metis_output(grp, metis_output)
  if breakup:
    # Heuristic to cut the wide spread clusters
    break_up_threshold = get_break_up_threshold(plc)
    # Get a measure to identify small clusters.
    # This is a number that's 1/4th of average nodes per group.
    merge_threshold = plc.num_nodes() // grp.num_groups() // 4
    closeness = break_up_threshold / 2.0
    break_up_and_merge(grp, break_up_threshold, merge_threshold, closeness)
  else:
    logging.info('Skipping break up of clusters.')
  logging.info('writing netlist: %s', filename)
  grp.write_grouped_netlist(filename)
  final_groups_file = os.path.join(output_dir, 'groups.final')
  write_final_groupings(plc, grp, final_groups_file)

  plc_file = os.path.join(output_dir, 'initial.plc')
  # We need to propagate the original size info.
  orig_canvas_width, orig_canvas_height = plc.get_canvas_width_height()
  orig_grid_cols, orig_grid_rows = plc.get_grid_num_columns_rows()
  # Propagate original attributes (sizes, blockages).
  new_plc = placement_util.create_placement_cost_using_common_arguments(
      netlist_file=filename,
      canvas_width=orig_canvas_width,
      canvas_height=orig_canvas_height,
      grid_cols=orig_grid_cols,
      grid_rows=orig_grid_rows,
      blockages=plc.get_blockages())

  extra_info = 'Original source netlist with standard cells: {}\n'.format(
      plc.get_source_filename())
  extra_info += 'Groups file: {}\n'.format(final_groups_file)
  extra_info += worst_spread_metrics_log(grp)

  placement_util.save_placement_with_info(new_plc, plc_file, extra_info)

  logging.info('Placement file : %s, WL: %f, cong: %f}', plc_file,
               new_plc.get_wirelength(), new_plc.get_congestion_cost())
  return filename, plc_file


def get_new_output_dir(ngrps: int, hopts: Any) -> str:
  return 'g{}_ub{}_nruns{}_c{}_r{}_v{}_rc{}'.format(ngrps, hopts.ub_factor,
                                                    hopts.n_runs, hopts.c_type,
                                                    hopts.r_type, hopts.v_cycle,
                                                    hopts.reconst)


def run_with_default_hmetis_options(
    plc: plc_client.PlacementCost, num_groups: int, fixed_logic_levels: int,
    cell_area_utilization: float, breakup: bool, output_dir: str,
    netlist_file: str) -> Optional[Tuple[str, str]]:
  """Runs hMETIS with default options."""

  @dataclasses.dataclass
  class HmetisOptions:
    ub_factor: int
    n_runs: int
    c_type: int
    r_type: int
    v_cycle: int
    reconst: int
    dbglvl: int

  hopts = HmetisOptions(
      ub_factor=5,
      n_runs=10,
      c_type=5,
      r_type=3,
      v_cycle=3,
      reconst=1,
      dbglvl=0)
  new_output_dir = os.path.join(output_dir,
                                get_new_output_dir(num_groups, hopts))
  return partition_netlist(plc, num_groups, fixed_logic_levels,
                           cell_area_utilization, breakup, new_output_dir,
                           hopts, netlist_file)


def update_groups_using_metis_output(grp: grouping.Grouping,
                                     metis_out_file: str):
  """Sets each node's group number according to the metis output file."""
  metis_groups = read_metis_out_file(metis_out_file)
  num_fixed_groups = grp.num_groups()
  for node_index, group_index in metis_groups.items():
    existing_group = grp.get_node_group(node_index)
    if existing_group > -1:
      if group_index != existing_group:
        raise RuntimeError(
            f'group_index {group_index} is not equal to existing_group {existing_group}'
        )
      continue
    if group_index > -1:
      grp.set_node_group(node_index, group_index + num_fixed_groups)
    else:
      grp.ungroup_node(node_index)


def read_metis_out_file(filename: str) -> Dict[int, int]:
  metis_groups = sortedcontainers.SortedDict()
  with open(filename, 'r') as infile:
    node_index = 0
    for line in infile:
      metis_groups[node_index] = int(line)
      node_index += 1
  return metis_groups


def write_final_groupings(plc: plc_client.PlacementCost, grp: grouping.Grouping,
                          filename: str) -> None:
  with open(filename, 'w') as outfile:
    for node_index in range(plc.num_nodes()):
      grpindex = grp.get_node_group(node_index)
      outfile.write(str(grpindex) + '\n')


def worst_spread_metrics_log(grp: grouping.Grouping,
                             num_worst: int = 10) -> str:
  """Generates the report for the worst group spreads.

  Args:
    grp: Grouping object.
    num_worst: number of worst grouping to report.

  Returns:
    A report of the worst group spreads.
  """
  grp_spread = dict()
  for grp_id in grp.group_ids():
    grp_spread[grp_id] = grp.spread_metric(grp_id)
  sorted_list = sorted(
      list(grp_spread.items()), key=lambda kv: (kv[1], kv[0]), reverse=True)
  result = 'worst {} spread\n'.format(num_worst)
  for v in sorted_list[0:num_worst]:
    result += 'grp: {} - spread: {}\n'.format(v[0], v[1])
  return result


def get_highest_group_index(grp: grouping.Grouping) -> int:
  return max(grp.group_ids())


def break_up_and_merge(grp: grouping.Grouping, break_up_threshold: float,
                       merge_threshold: int, closeness: float) -> None:
  """Break up groups that are spannig a large area, and merge small groups.

  Args:
    grp: Grouping object.
    break_up_threshold: Break up a group in x an y directions if it spans a
      larger distance than this threshold.
    merge_threshold: Try merging a group with another close by adjacent group if
      it has fewer nodes than this amount.
    closeness: Merge small groups if they are close enough by this amount.
  """
  logging.info('before break up:')
  logging.info('num groups: %d', grp.num_groups())
  logging.info(worst_spread_metrics_log(grp))
  logging.info('breaking up groups spanning more than %.3f distance',
               break_up_threshold)
  grp.breakup_groups(break_up_threshold)
  logging.info('after break up:')
  logging.info('num groups: %d', grp.num_groups())
  logging.info(worst_spread_metrics_log(grp))

  if merge_threshold <= 0:
    return
  logging.info(
      'merging groups with smaller than %d nodes to close by (%.3f microns)'
      ' connected groups', merge_threshold, closeness)
  while not grp.merge_small_adj_close_groups(merge_threshold, closeness):
    pass
  logging.info('after merge:')
  logging.info('num groups: %d', grp.num_groups())
  logging.info(worst_spread_metrics_log(grp))


def print_cost_info(plc: plc_client.PlacementCost) -> None:
  logging.info('Wirelength: %f', plc.get_wirelength())
  logging.info('Wirelength cost: %f', plc.get_cost())
  logging.info('Congestion cost: %f', plc.get_congestion_cost())
  logging.info('Overlap cost: %f', plc.get_overlap_cost())


def get_grid_cell_width_height(
    plc: plc_client.PlacementCost) -> Tuple[float, float]:
  canvas_width, canvas_height = plc.get_canvas_width_height()
  cols, rows = plc.get_grid_num_columns_rows()
  return canvas_width / cols, canvas_height / rows


def add_blockage(plc: plc_client.PlacementCost,
                 block_name: str,
                 blockage_cl: Optional[str] = None) -> None:
  """Adds blockage info to the plc.

  Args:
    plc: A placement cost object.
    block_name: Fp name of the netlist.
    blockage_cl: Optional cl number to read the info from.
  """
  blockage_file = FLAGS.blockage_file

  if blockage_file:
    w, h = plc.get_canvas_width_height()
    blockages = placement_util.extract_blockages_from_tcl(
        blockage_file, block_name, w, h, FLAGS.is_rectilinear)
    for blockage in blockages:
      logging.info('Blockage: %s', blockage)
      plc.create_blockage(*blockage)


def select_grid_size(plc: plc_client.PlacementCost) -> None:
  """Selects grid size."""
  logging.info('Try auto grid selection...')
  cols, rows = grid_size_selection.get_grid_suggestion(plc)
  if cols and rows:
    plc.set_placement_grid(cols, rows)
    # Force FLAG settings, as well.
    logging.info('Grid columns: %d, rows: %d, total grid cells: %d.', cols,
                 rows, cols * rows)
  else:
    logging.error('The auto grid selection failed.')
    raise RuntimeError('Failed to select a grid size.')


def group_stdcells(
    netlist_file: str,
    output_dir: str,
    block_name: str,
    create_placement_cost_fn: Callable[..., plc_client.PlacementCost] = (
        placement_util.create_placement_cost_using_common_arguments),
    blockage_cl: Optional[str] = None,
) -> Tuple[plc_client.PlacementCost, str]:
  """Groups stdcells and set grid size.

  Args:
    netlist_file: Path to the original (non-clustred) netlist file.
    output_dir: Path to a directory to store output files.
    block_name: Fp block name.
    create_placement_cost_fn: function to make plc_client.
    blockage_cl: Optional cl to read blockage info from.

  Returns:
    A tuple of a plc object of the clustred netlist and path to the plc file of
    the same clustred netlist.
  """

  plc = create_placement_cost_fn(netlist_file=netlist_file)

  logging.info('disconnecting high fanout nets')
  placement_util.disconnect_high_fanout_nets(plc)
  logging.info('Number of macros: %s',
               placement_util.num_nodes_of_type(plc, 'MACRO'))
  logging.info('Number of stdcells: %s',
               placement_util.num_nodes_of_type(plc, 'STDCELL'))
  logging.info('Number of ports: %s',
               placement_util.num_nodes_of_type(plc, 'PORT'))

  # We need blockage information here, since grid size selection can be
  # affected by blockages. Also, this is the place where we inject
  # blockage information to the downstream .plc files.
  if FLAGS.extract_blockages:
    add_blockage(plc, block_name, blockage_cl)

  if FLAGS.auto_grid_size:
    select_grid_size(plc)

  cols, rows = plc.get_grid_num_columns_rows()
  logging.info('using %d columns and %d rows.', cols, rows)
  output_dir = os.path.join(output_dir, '{}cols_{}rows'.format(cols, rows))

  part_and_plc_files = run_with_default_hmetis_options(
      plc, FLAGS.num_groups, FLAGS.fixed_logic_levels,
      FLAGS.cell_area_utilization, FLAGS.breakup, output_dir, netlist_file)
  assert part_and_plc_files, 'groupping failed.'
  part_file, plc_file = part_and_plc_files

  logging.info('Partitioned netlist: %s, plc file: %s', part_file, plc_file)

  logging.info('Previous costs:')
  print_cost_info(plc)

  grouped_plc = create_placement_cost_fn(
      netlist_file=part_file, init_placement=plc_file)

  logging.info('Costs for partitioned netlist:')
  print_cost_info(grouped_plc)

  legalized_placement = os.path.join(os.path.dirname(plc_file), 'legalized.plc')

  placement_util.legalize_placement(grouped_plc)
  placement_util.save_placement_with_info(
      grouped_plc, legalized_placement,
      'Original file : {}\nInitial placement : {}\n'.format(
          netlist_file, plc_file))
  logging.info('Saved legalized placement : %s', legalized_placement)
  logging.info('Costs after legalization:')
  print_cost_info(grouped_plc)

  return grouped_plc, legalized_placement
