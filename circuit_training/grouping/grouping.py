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
"""Grouping Class."""
import collections
import copy
import math
from typing import List, Tuple, Dict, Union

import sortedcontainers
import tensorflow as tf

from circuit_training.grouping import meta_netlist_data_structure as mnds
from google.protobuf import text_format
# Internal gfile dependencies

# Used as a default value for non exist index.
_NON_EXIST_INDEX = -1

# Used as a default value of the location for node.
_BAD_PAIR = (-1, -1)

# Used as the minimum area.
_EPSILON = 1e-9


class Grouping:
  """Groups cells in a meta_netlist.

  Grouping class is used to group (cluster) standard cells in a meta_netlist.
  It has interfaces to write out hMetis files.
  """

  def __init__(self,
               meta_netlist: mnds.MetaNetlist,
               max_group_id: int = 0,
               cell_area_utilization: float = 0.5) -> None:
    """Initializes the grouping class.

    Args:
      meta_netlist: The parsed MetaNetlist data.
      max_group_id: The max group id.
      cell_area_utilization: Cell arae utilization.
    """
    self._meta_netlist = meta_netlist

    # Key: group id, Value: set of node indices.
    self._node_groups = collections.defaultdict(sortedcontainers.SortedSet)

    # Holds the groups information. Key: node_index, Value: group id.
    self._node_group_map = sortedcontainers.SortedDict()

    self._max_group_id = max_group_id

    self._cell_area_utilization = cell_area_utilization

  def reset_groups(self) -> None:
    """Reset all groups."""
    self._node_groups.clear()
    self._node_group_map.clear()
    self._max_group_id = 0

  def set_cell_area_utilization(self, ratio: float) -> None:
    """Sets cell area utilization."""
    self._cell_area_utilization = ratio

  def get_side(self, x: float, y: float, width: float,
               height: float) -> mnds.Side:
    r"""Returns which side of the canvas area coordinate is.

    Uses two lines drawn across the corners diagonally.
    y = m*x is the line connecting lower left and upper right corners.
    y = height - m*x is the line connecting upper left and lower right corners.
       \     /
        \   /
         \ /
          X
         / \
        /   \
       /     \

    Args:
      x: the x coordinate of the node.
      y: The y coordinate of the node.
      width: The canvas width.
      height: The canvas height.
    """
    mx = x * height / width
    if y > mx:
      if y > height - mx:
        return mnds.Side.TOP
      else:
        return mnds.Side.LEFT
    else:
      if y > height - mx:
        return mnds.Side.RIGHT
      else:
        return mnds.Side.BOTTOM

  def group_ids(self) -> List[int]:
    """Groups ids."""
    return sorted(list(self._node_groups))

  def setup_fixed_groups(self, logic_levels_to_traverse: int) -> None:
    """Setup the fixed groups.

    Assigns nodes in the graph to a group, In hMetis terms it becomes a
    fixed group, and they are not moved around to other groups.
    Fixed groups are groups of stdcells that are connected (within a certain
    logic level distance) to:
    1. Same macro's pins
    2. I/O's that are within close proximity of each other

    Args:
      logic_levels_to_traverse: A set of logic cells that are connected to
        either outputs or inputs of a starting logic cell.
    """
    group_index = 0
    # Goes through each macro, put each macro's pins into a separate group.
    for netlist_node in self._meta_netlist.node:
      if (netlist_node.type == mnds.Type.MACRO and not netlist_node.soft_macro):
        for pin_ind in netlist_node.output_indices:
          self.set_node_group(pin_ind, group_index)

        for pin_ind in netlist_node.input_indices:
          self.set_node_group(pin_ind, group_index)
        group_index += 1

    # Goes through I/O's.
    # Most netlists will not have side constraints for ports, infer from the
    # location and put each side's ports in a vector with their x or y
    # coordinate so that we can sort them and group them by proximity.
    canvas_width = self._meta_netlist.canvas.dimension.width
    canvas_height = self._meta_netlist.canvas.dimension.height

    ports_at_side = [[] for _ in range(4)]

    for netlist_node in self._meta_netlist.node:
      # If a port is not placed (rare condition) we cannot group them
      # by proximity.
      if netlist_node.type != mnds.Type.PORT or netlist_node.coord is None:
        continue

      side = self.get_side(netlist_node.coord.x, netlist_node.coord.y,
                           canvas_width, canvas_height)
      if side in {mnds.Side.LEFT, mnds.Side.RIGHT}:
        key_coord = netlist_node.coord.y
      else:
        key_coord = netlist_node.coord.x

      ports_at_side[side].append(
          mnds.CoordIndex(coord=key_coord, id=netlist_node.id))

    num_cols = self._meta_netlist.canvas.num_columns
    num_rows = self._meta_netlist.canvas.num_rows

    for side in range(4):
      # Need to increase group index when starting a new side if previous
      # side was not empty.
      if side > 0 and ports_at_side[side - 1]:
        group_index += 1

      if not ports_at_side[side]:
        continue

      if side in {mnds.Side.LEFT, mnds.Side.RIGHT}:
        group_distance = canvas_height / num_rows
      else:
        group_distance = canvas_width / num_cols

      # Order ports based on the key coordinate.
      ports_at_side[side].sort(key=lambda x: x.coord)
      last_coord = ports_at_side[side][0].coord
      for coord_index_pair in ports_at_side[side]:
        this_coord = coord_index_pair.coord
        if this_coord - last_coord > group_distance:
          group_index += 1
          last_coord = this_coord
        self.set_node_group(coord_index_pair.id, group_index)

    for _ in range(logic_levels_to_traverse):
      self.expand_group_by_one_level()

  def expand_group_by_one_level(self):
    """Expands the group by one level.

    Traverses one level of logic hierarchy from the existing groups, and assign
    those traversed stdcells to the groups.
    """
    initial_data = copy.deepcopy(self._node_group_map)

    def _insert_into_group(inout_index, group_index):
      if inout_index in self._node_group_map or self._meta_netlist.node[
          inout_index].type != mnds.Type.STDCELL:
        return
      self.set_node_group(inout_index, group_index)

    for node_index, group_index in initial_data.items():
      for out_index in self.get_fan_outs_of_node(node_index):
        _insert_into_group(out_index, group_index)

      for in_index in self.get_fan_ins_of_node(node_index):
        _insert_into_group(in_index, group_index)

  def get_fan_outs_of_node(self, node_index: int) -> List[int]:
    """Gets the output_indices of a node."""
    if node_index < 0 or node_index >= len(self._meta_netlist.node):
      return []

    return self._meta_netlist.node[node_index].output_indices

  def get_fan_ins_of_node(self, node_index: int) -> List[int]:
    """Gets the input_indices of a node."""
    if node_index < 0 or node_index >= len(self._meta_netlist.node):
      return []

    return self._meta_netlist.node[node_index].input_indices

  def num_groups(self) -> int:
    """The number of groups."""
    return len(self._node_groups)

  def ungroup_node(self, node_index: int) -> None:
    """Ungroups a node."""
    if node_index not in self._node_group_map:
      return

    group_id = self._node_group_map[node_index]
    self._node_group_map.pop(node_index)

    if group_id not in self._node_groups:
      return

    self._node_groups[group_id].remove(node_index)

    # Removes group key in the node_groups if it is empty.
    if not self._node_groups[group_id]:
      self._node_groups.pop(group_id)

  def set_node_group(self, node_index: int, group_index: int) -> None:
    """Set a node to a group."""
    self.ungroup_node(node_index)
    if group_index < 0:
      return

    self._node_group_map[node_index] = group_index
    self._node_groups[group_index].add(node_index)

    if self._max_group_id < group_index:
      self._max_group_id = group_index

  def get_node_group(self, node_index: int) -> int:
    """Gets the group id for a node.

    Args:
      node_index: The input node index.

    Returns:
      The group id of the node. Return _NON_EXIST_INDEX(-1) if it is not found.
    """
    return self._node_group_map.get(node_index, _NON_EXIST_INDEX)

  def write_metis_file(self, file_path: str) -> None:
    """Writes metis groups to file."""
    num_lines = 0
    lines = []
    for node in self._meta_netlist.node:
      if node.type == mnds.Type.MACRO or not node.output_indices:
        continue

      # Adding 1 to the indices, hMetis accepts node indices from 1 to n.
      # Every line lists the connected node indices.
      lines.append(f"{node.id + 1}")

      for output_index in node.output_indices:
        lines.append(f" {output_index + 1}")

      lines.append("\n")
      num_lines += 1

    header_line = f"{num_lines} {len(self._meta_netlist.node)}\n"

    with open(file_path, "w") as f:
      f.write("".join([header_line] + lines))

  def write_metis_fix_file(self, file_path: str) -> None:
    """Writes out the group fix file for metis."""
    lines = []
    for i, _ in enumerate(self._meta_netlist.node):
      group_id = self._node_group_map.get(i, _NON_EXIST_INDEX)
      lines.append(f"{group_id}\n")

    with open(file_path, "w") as f:
      f.write("".join(lines))

  def get_node_outputs(self, node_index: int) -> Dict[int, float]:
    """Gets node outputs."""
    current_group = -1
    if self._meta_netlist.node[node_index].type == mnds.Type.STDCELL:
      # current_group is relevant only if this node is a standard cell.
      current_group = self.get_node_group(node_index)

    grp_fanouts = sortedcontainers.SortedSet()
    for out_index in self.get_fan_outs_of_node(node_index):
      node_or_grp_index = out_index
      if self._meta_netlist.node[out_index].type == mnds.Type.STDCELL:
        grp_no = self.get_node_group(out_index)
        if grp_no < 0:
          continue

        if grp_no == current_group:
          # Do not add outputs if the driven stdcell is already in the same
          # group
          continue

        # positive numbers are for node indices, groups are encoded as negative
        # numbers (requires a -1 to distinguish node 0 and group 0.
        node_or_grp_index = -grp_no - 1
      grp_fanouts.add(node_or_grp_index)

    weight = self._meta_netlist.node[node_index].weight

    node_fanout = sortedcontainers.SortedDict()
    for ind in grp_fanouts:
      if ind in node_fanout:
        node_fanout[ind] += weight
      else:
        node_fanout[ind] = weight

    return node_fanout

  def get_new_node_name(self, index: int) -> str:
    """Gets a new node name."""
    if index < 0:
      return f"Grp_{-1 - index}/Pinput"

    return self._meta_netlist.node[index].name

  def group_area(self, group_index: int) -> float:
    """Gets group arae."""
    area = 0
    for node_index in self._node_groups[group_index]:
      if self._meta_netlist.node[node_index].type != mnds.Type.STDCELL:
        continue
      width, height = self.get_node_width_height(node_index)
      area += width * height
    return area

  def write_as_macro(self, group_no: int,
                     graph_def: tf.compat.v1.GraphDef) -> None:
    """Appends the macro definition to protobuf."""
    group_vect_p = self._node_groups.get(group_no, None)
    if group_vect_p is None or not group_vect_p:
      return

    macro_name = f"Grp_{group_no}"
    area = self.group_area(group_no)
    # Bloat group area to achieve desired utilization.
    area = area / self._cell_area_utilization
    x_coord, y_coord = self.group_coordinates(group_no)
    # Setting the group width to grid width.
    group_width = (
        self._meta_netlist.canvas.dimension.width /
        self._meta_netlist.canvas.num_columns)
    group_height = area / group_width
    new_node = graph_def.node.add()
    new_node.name = macro_name
    self.add_attr(new_node, "type", "macro")
    self.add_attr(new_node, "width", group_width)
    self.add_attr(new_node, "height", group_height)
    self.add_attr(new_node, "x", x_coord)
    self.add_attr(new_node, "y", y_coord)

    single_fanouts = sortedcontainers.SortedDict()
    pindex = 0

    # Handling multi fanout nets crossing group boundaries.
    for node_index in self._node_groups[group_no]:
      if self._meta_netlist.node[node_index].type != mnds.Type.STDCELL:
        continue

      node_fanout = self.get_node_outputs(node_index)
      if not node_fanout:
        continue

      if len(node_fanout) == 1:
        key, value = list(node_fanout.items())[0]
        if key in single_fanouts:
          single_fanouts[key] += value
        else:
          single_fanouts[key] = value
        continue

      pin_name = f"Grp_{group_no}/Poutput_multi_{pindex}"
      pindex += 1
      new_node = graph_def.node.add()
      new_node.name = pin_name
      self.add_attr(new_node, "type", "macro_pin")
      self.add_attr(new_node, "macro_name", macro_name)
      self.add_attr(new_node, "x", x_coord)
      self.add_attr(new_node, "y", y_coord)
      self.add_attr(new_node, "x_offset", 0.0)
      self.add_attr(new_node, "y_offset", 0.0)

      weight = list(node_fanout.items())[0][1]
      if weight != 1.0:
        self.add_attr(new_node, "weight", float(weight))

      for driven_index in node_fanout:
        driven_name = self.get_new_node_name(driven_index)
        new_node.input.append(driven_name)

    pindex = 0
    # Create single, (probably multi weight) output pins.
    for driven_index, weight in single_fanouts.items():
      driven_name = self.get_new_node_name(driven_index)
      pin_name = f"Grp_{group_no}/Poutput_single_{pindex}"
      pindex += 1
      new_node = graph_def.node.add()
      new_node.name = pin_name
      new_node.input.append(driven_name)
      self.add_attr(new_node, "type", "macro_pin")
      self.add_attr(new_node, "macro_name", macro_name)

      if weight != 1.0:
        self.add_attr(new_node, "weight", float(weight))

      self.add_attr(new_node, "x", x_coord)
      self.add_attr(new_node, "y", y_coord)
      self.add_attr(new_node, "x_offset", 0.0)
      self.add_attr(new_node, "y_offset", 0.0)

    # Create an input pin.
    macro_input_pin_name = f"{macro_name}/Pinput"
    new_node = graph_def.node.add()
    new_node.name = macro_input_pin_name
    self.add_attr(new_node, "type", "macro_pin")
    self.add_attr(new_node, "macro_name", macro_name)
    self.add_attr(new_node, "x", x_coord)
    self.add_attr(new_node, "y", y_coord)
    self.add_attr(new_node, "x_offset", 0.0)
    self.add_attr(new_node, "y_offset", 0.0)

  def add_attr(self, node: tf.compat.v1.NodeDef, attr_name: str,
               attr_value: Union[str, float]) -> None:
    """Adds attributes to the node."""
    if isinstance(attr_value, float):
      node.attr[attr_name].f = attr_value
    else:
      node.attr[attr_name].placeholder = attr_value

  def write_grouped_netlist(self, file_path: str) -> None:
    """Writes out a new tensorflow metagraph protobuf file."""
    groups_to_print = sortedcontainers.SortedSet()
    graph_def = tf.compat.v1.GraphDef()
    metadata_node = graph_def.node.add()
    metadata_node.name = "__metadata__"
    metadata_node.attr[
        "soft_macro_area_bloating_ratio"].f = 1.0 / self._cell_area_utilization

    for node in self._meta_netlist.node:
      node_index = node.id
      if node.type == mnds.Type.STDCELL:
        node_group = self.get_node_group(node_index)
        if node_group > -1:
          groups_to_print.add(node_group)
          continue

      new_node = graph_def.node.add()
      new_node.name = node.name
      if node.type != mnds.Type.MACRO:
        node_fanout = self.get_node_outputs(node_index)
        for driven_index in node_fanout:
          new_node.input.append(self.get_new_node_name(driven_index))

        if node.weight != 1.0:
          self.add_attr(new_node, "weight", float(node.weight))

      self.add_attr(new_node, "type", node.type.name)

      if node.coord is not None:
        self.add_attr(new_node, "x", node.coord.x)
        self.add_attr(new_node, "y", node.coord.y)

      if node.offset is not None:
        self.add_attr(new_node, "x_offset", node.offset.x)
        self.add_attr(new_node, "y_offset", node.offset.y)

      if node.dimension is not None:
        self.add_attr(new_node, "width", node.dimension.width)
        self.add_attr(new_node, "height", node.dimension.height)

      if node.constraint is not None:
        self.add_attr(new_node, "side", node.constraint.side.name)

      if node.type == mnds.Type.MACRO_PIN:
        self.add_attr(new_node, "macro_name",
                      self._meta_netlist.node[node.ref_node_id].name)

      if node.type == mnds.Type.MACRO:
        if node.orientation is None:
          self.add_attr(new_node, "orientation", mnds.Orientation.N.name)
        else:
          self.add_attr(new_node, "orientation", mnds.Orientation.name)

    for group_no in groups_to_print:
      self.write_as_macro(group_no, graph_def)

    with open(file_path, "w") as f:
      f.write(text_format.MessageToString(graph_def))

  def get_node_location(self, node_index: int) -> Tuple[float, float]:
    """Gets location of a node."""
    if node_index < 0 or node_index >= len(self._meta_netlist.node):
      return _BAD_PAIR

    node = self._meta_netlist.node[node_index]
    if node.coord is None:
      return _BAD_PAIR

    return node.coord.x, node.coord.y

  def get_node_width_height(self, node_index: int) -> Tuple[float, float]:
    """Gets width and height of a node."""
    if node_index < 0 or node_index >= len(self._meta_netlist.node):
      return _BAD_PAIR

    node = self._meta_netlist.node[node_index]
    if node.dimension is None:
      return _BAD_PAIR

    return node.dimension.width, node.dimension.height

  def group_coordinates(self, group_index: int) -> Tuple[float, float]:
    """Returns the center of mass coordinates for the stdcells in the group."""
    x_weighted_sum = 0
    y_weighted_sum = 0
    divisor = 0
    if group_index not in self._node_groups:
      return 0.0, 0.0

    for node_index in self._node_groups[group_index]:
      if self._meta_netlist.node[node_index].type != mnds.Type.STDCELL:
        continue

      x, y = self.get_node_location(node_index)
      width, height = self.get_node_width_height(node_index)
      area = width * height
      x_weighted_sum += x * area
      y_weighted_sum += y * area
      divisor += area

    if divisor < _EPSILON:
      return 0.0, 0.0

    return x_weighted_sum / divisor, y_weighted_sum / divisor

  def spread_metric(self, group_id: int) -> float:
    """Returns how much the stdcells in a group are spread apart."""
    group_vect_p = self._node_groups.get(group_id, None)

    if group_vect_p is None or not group_vect_p:
      return 0

    c_x, c_y = self.group_coordinates(group_id)
    xsqr_sum = 0
    ysqr_sum = 0
    for node_index in group_vect_p:
      node = self._meta_netlist.node[node_index]
      if node.coord is not None and node.type == mnds.Type.STDCELL:
        xdiff = c_x - node.coord.x
        xsqr_sum += xdiff * xdiff
        ydiff = c_y - node.coord.y
        ysqr_sum += ydiff * ydiff

    spread_metric = math.sqrt(
        math.sqrt(xsqr_sum) * math.sqrt(ysqr_sum)) * len(group_vect_p)
    return spread_metric

  def is_close(self, a: Tuple[float, float], b: Tuple[float, float],
               distance: float) -> bool:
    """Returns true if manhattan distance of two points is close."""
    xa, ya = a
    xb, yb = b
    return (abs(xa - xb) + abs(ya - yb)) <= distance

  def merge_small_adj_close_groups(self, max_num_nodes: int,
                                   distance: float) -> bool:
    """Merges small adjacency groups.

    Merges small groups to the most adjacent group if they are within
    a certain distance.  The function returns false if there may be a need for
    another call to merge. Ideally this should be called repeatedly until it
    returns true. True means there are no more possible merges.

    Note that this should be created from scratch each time, since the groups
    may have changed.

    Args:
      max_num_nodes: The maximum number of nodes.
      distance: The distance used to determine if two nodes are close or not.

    Returns:
      True if there are no more possible merges, False otherwise.
    """
    adj_matrix = []

    total_num_groups = self._max_group_id + 1

    adj_matrix = [0] * total_num_groups * total_num_groups
    for node_index, group_id in self._node_group_map.items():
      groups_in_net = set()
      groups_in_net.add(group_id)
      for out_index in self.get_fan_outs_of_node(node_index):
        other_group_id = self._node_group_map.get(out_index, _NON_EXIST_INDEX)
        if other_group_id > _NON_EXIST_INDEX:
          groups_in_net.add(other_group_id)

      for i in groups_in_net:
        for j in groups_in_net:
          if i != j:
            adj_matrix[i * total_num_groups + j] += 1

    group_ids = self.group_ids()

    # Calculate group coords for fast multiple lookups.
    group_coords = [None] * total_num_groups
    for group_id in group_ids:
      group_coords[group_id] = self.group_coordinates(group_id)

    finished = True
    for group_id in group_ids:
      if len(self._node_groups[group_id]) > max_num_nodes:
        continue

      # Going through the small sized groups, find the highest adjacency group
      # within the given distance.
      max_adj_grp = -1
      max_adj = 0
      for i in range(total_num_groups):
        adj = adj_matrix[i * total_num_groups + group_id]
        if i == group_id or adj == 0:
          continue

        # Check the proximity, save the index of the group with the highest
        # adjacency.
        if self.is_close(group_coords[i], group_coords[group_id], distance):
          if max_adj < adj:
            max_adj = adj
            max_adj_grp = i

      if max_adj_grp > -1:
        # Found one group to merge to. Make a copy of _node_groups item,
        # since SetNodeGroup might invalidate the iterators.
        nodes_copy = copy.deepcopy(self._node_groups[group_id])
        for node_id in nodes_copy:
          self.set_node_group(node_id, max_adj_grp)

        if len(self._node_groups[max_adj_grp]) <= max_num_nodes:
          # This new merged group's size is smaller than max_num_nodes.
          # We need to signal the caller that another pass is needed.
          finished = False

    return finished

  def get_bounding_box(self, group_id: int) -> mnds.BoundingBox:
    """Gets bounding box."""
    group_vect_p = self._node_groups.get(group_id, _NON_EXIST_INDEX)
    bbox = mnds.BoundingBox(minx=1e10, miny=1e10, maxx=-1e10, maxy=-1e10)
    if group_vect_p == _NON_EXIST_INDEX or not group_vect_p:
      return bbox

    for node_index in group_vect_p:
      node = self._meta_netlist.node[node_index]
      if node.coord is not None:
        x = node.coord.x
        y = node.coord.y
        bbox.minx = min(bbox.minx, x)
        bbox.maxx = max(bbox.maxx, x)
        bbox.miny = min(bbox.miny, y)
        bbox.maxy = max(bbox.maxy, y)

    return bbox

  def x_bucket(self, x: float, box: mnds.BoundingBox, cut_size: float,
               center: Tuple[float, float]) -> int:
    """Gets x bucket."""
    if box.maxx - box.minx < cut_size:
      return 0

    x_center, _ = center

    if x > x_center:
      return int(0.5 + (x - x_center) / cut_size)

    return int(-0.5 + (x - x_center) / cut_size)

  def y_bucket(self, y: float, box: mnds.BoundingBox, cut_size: float,
               center: Tuple[float, float]) -> int:
    """Gets y bucket."""
    if box.maxy - box.miny < cut_size:
      return 0

    _, y_center = center
    if y > y_center:
      return int(0.5 + (y - y_center) / cut_size)

    return int(-0.5 + (y - y_center) / cut_size)

  def breakup_groups(self, threshold: float):
    """Breaks up groups that span a distance larger than threshold."""
    for group_id in self.group_ids():
      grp_bbox = self.get_bounding_box(group_id)
      if (grp_bbox.maxx - grp_bbox.minx >
          threshold) or (grp_bbox.maxy - grp_bbox.miny > threshold):
        coord = self.group_coordinates(group_id)
        gcell_vs_new_group = sortedcontainers.SortedDict()
        # Must make a copy of node_groups_, since SetNodeGroup modifies it.
        nodes_copy = copy.deepcopy(self._node_groups[group_id])
        for node_index in nodes_copy:
          # Bucketize each node in 2-d based on XBucket, YBucket, so that
          # each bucket xy span will be less than threshold. The nodes in
          # the center bucket will not be moved to a new group.
          node = self._meta_netlist.node[node_index]
          xb = self.x_bucket(node.coord.x, grp_bbox, threshold, coord)
          yb = self.y_bucket(node.coord.y, grp_bbox, threshold, coord)

          if xb == 0 and yb == 0:
            continue

          new_group_id = gcell_vs_new_group.get((xb, yb), _NON_EXIST_INDEX)

          if new_group_id == _NON_EXIST_INDEX:
            new_group_id = self._max_group_id + 1
            gcell_vs_new_group[(xb, yb)] = new_group_id

          self.set_node_group(node_index, new_group_id)
