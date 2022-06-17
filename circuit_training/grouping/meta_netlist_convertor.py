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
"""Convert functions for MetaNetlist."""
import itertools
from typing import Any, Dict

from absl import logging
import numpy as np
import tensorflow as tf

from circuit_training.grouping import meta_netlist_data_structure as mnds
from google.protobuf import text_format
# Internal gfile dependencies

# Default number of columns and rows for canvas.
_DEFAULT_NUM_COLS_ROWS = 10

# The minimum canvas side length.
_MIN_SIZE = 1e-3

# The maximum density for the canvas.
_MAX_DENSITY = 0.6

# The threholsd for high fanout node. If above this number, it logs the node
# name.
_HIGH_FANOUT = 100


def read_attr(node: tf.compat.v1.NodeDef, attr_name: str) -> Any:
  """Read attribute from node.

  Args:
    node: Input node.
    attr_name: Attributes name.

  Returns:
    The value of the attribute in the node. If the attribute is not found in the
    node, returns None.
  """
  attr = node.attr.get(attr_name, None)

  if attr is None:
    return None

  if attr.WhichOneof("value") is None:
    return None

  return getattr(attr, attr.WhichOneof("value"))


def translate_node(node: tf.compat.v1.NodeDef,
                   name_to_id_map: Dict[str, int]) -> mnds.NetlistNode:
  """Translate a tensorflow node to a NetlistNode.

  Args:
    node: Tensorflow node.
    name_to_id_map: A map from name to id. The id will be used in assign the id
      in the NetlistNode.

  Returns:
    A converted netlistNode.

  Raises:
    ValueError: If the expected type is wrong or attributes assigned to the
      wrong node.
    KeyError: If the name is not found in the name_to_id_map,  or the type,
      orientation cannot be found in the corresponding enum list.
  """
  # It raises KeyError if node.name is not found in the name_to_id_map.
  node_ind = name_to_id_map[node.name]

  netlist_node = mnds.NetlistNode()
  netlist_node.id = node_ind
  netlist_node.name = node.name

  uniq_outputs = set()
  for node_name in node.input:
    # It raises KeyError if node_name is not found in the name_to_id_map.
    node_ind = name_to_id_map[node_name]

    if node_ind not in uniq_outputs:
      netlist_node.output_indices.append(node_ind)
    uniq_outputs.add(node_ind)

  node_type = read_attr(node, "type")
  if node_type is not None:
    node_type = mnds.Type[node_type.upper()]
    netlist_node.type = node_type
    if node_type == mnds.Type.MACRO:
      netlist_node.soft_macro = node.name.startswith("Grp_")
  else:
    raise ValueError(
        f"Required attribute 'type' not found for node: {node.name}")

  orientation = read_attr(node, "orientation")
  if orientation is not None:
    if netlist_node.type != mnds.Type.MACRO:
      raise ValueError("'orientation' attribute is only for macros.")
    netlist_node.orientation = mnds.Orientation[orientation.upper()]

  x = read_attr(node, "x")
  y = read_attr(node, "y")

  if x is not None and y is not None:
    netlist_node.coord = mnds.Coord(x=x, y=y)

  x_offset = read_attr(node, "x_offset")
  y_offset = read_attr(node, "y_offset")

  if x_offset is not None and y_offset is not None:
    if netlist_node.type != mnds.Type.MACRO_PIN:
      raise ValueError(
          "'x_offset' and 'y_offset' attributes are only for macros_pin's.")
    netlist_node.offset = mnds.Offset(x=x_offset, y=y_offset)

  width = read_attr(node, "width")
  height = read_attr(node, "height")

  if width is not None and height is not None:
    netlist_node.dimension = mnds.Dimension(width=width, height=height)

  macro_name = read_attr(node, "macro_name")
  if macro_name is not None:
    if netlist_node.type != mnds.Type.MACRO_PIN:
      raise ValueError("'macro_name' attribute is only for macro_pins.")
    node_ind = name_to_id_map[macro_name]
    netlist_node.ref_node_id = node_ind

  side = read_attr(node, "side")
  if side is not None:
    if netlist_node.type != mnds.Type.PORT:
      raise ValueError("'side' attribute is only for ports.")
    netlist_node.constraint = mnds.Constraint(side=mnds.Side[side.upper()])

  weight = read_attr(node, "weight")
  if weight is not None:
    netlist_node.weight = weight
  else:
    netlist_node.weight = 1.0

  return netlist_node


def generate_canvas(total_area: float) -> mnds.Canvas:
  """Generates canvas from the total area.

  Args:
    total_area: total_area of the chip.

  Returns:
    Canvas with default values.
  """
  one_side = max(_MIN_SIZE, np.sqrt(total_area / _MAX_DENSITY))
  dimension = mnds.Dimension(width=one_side, height=one_side)
  return mnds.Canvas(
      dimension=dimension,
      num_rows=_DEFAULT_NUM_COLS_ROWS,
      num_columns=_DEFAULT_NUM_COLS_ROWS)


def place_macro_pin(netlist_node: mnds.NetlistNode,
                    netlist_node_macro: mnds.NetlistNode) -> None:
  """Places macro pin node.

  Places a macro pin using its ref macro's orientation and its offset. The
  changed is made inplace for netlist_node.

  Args:
    netlist_node: A netlist_node with MACRO_PIN type.
    netlist_node_macro: A netlist_node with MACRO type. It should be reference
      of the macro_pin node.

  Raises:
   ValueError: If the input netlist_node is not a type of MACRO_PIN node or
    netlist_node_macro is not a type of MACRO node.
  """
  if netlist_node.type != mnds.Type.MACRO_PIN:
    raise ValueError("Pleace make sure the input netlist_node is a type of "
                     "MACRO_PIN node.")

  if netlist_node_macro.type != mnds.Type.MACRO:
    raise ValueError(
        "Pleace make sure the input netlist_node_macro is a type of MACRO "
        "node.")

  x_offset = netlist_node.offset.x
  y_offset = netlist_node.offset.y
  x_offset_org = x_offset

  orientation = netlist_node_macro.orientation

  if orientation == mnds.Orientation.N:
    pass
  elif orientation == mnds.Orientation.FN:
    x_offset = -x_offset
  elif orientation == mnds.Orientation.S:
    x_offset = -x_offset
    y_offset = -y_offset
  elif orientation == mnds.Orientation.FS:
    y_offset = -y_offset
  elif orientation == mnds.Orientation.E:
    x_offset = y_offset
    y_offset = -x_offset_org
  elif orientation == mnds.Orientation.FE:
    x_offset = -y_offset
    y_offset = -x_offset_org
  elif orientation == mnds.Orientation.W:
    x_offset = -y_offset
    y_offset = x_offset_org
  elif orientation == mnds.Orientation.FW:
    x_offset = y_offset
    y_offset = x_offset_org

  x = netlist_node_macro.coord.x
  y = netlist_node_macro.coord.y

  netlist_node.coord.x = x + x_offset
  netlist_node.coord.y = y + y_offset


def convert_tfgraph_to_meta_netlist(
    netlist_tf_graph: tf.compat.v1.MetaGraphDef) -> mnds.MetaNetlist:
  """Converts the netlist in tf graph format to meta netlist.

  Args:
    netlist_tf_graph: The parsed netlist graph.

  Returns:
    A converted MetaNetlist.

  Raises:
    ValueError: If node names are not unique or certain fields are missing from
      the node definition.
  """
  name_to_id_map = {}
  ind = 0
  for node in netlist_tf_graph.graph_def.node:
    if node.name == "__metadata__":
      continue

    if not node.name:
      continue

    if node.name in name_to_id_map:
      raise ValueError(f"Node name not unique: {node.name}")

    name_to_id_map[node.name] = ind
    ind += 1

  id_to_name_map = {ind: name for name, ind in name_to_id_map.items()}

  netlist_node_list = []

  # Create netlist nodes, and translate attributes.
  for node in netlist_tf_graph.graph_def.node:
    if node.name == "__metadata__":
      continue

    netlist_node_list.append(translate_node(node, name_to_id_map))

  # Populate inputs of the output nodes, based on the outputs of the current
  # node.
  for netlist_node in netlist_node_list:
    if netlist_node.type == mnds.Type.MACRO:
      continue

    for out_ind in netlist_node.output_indices:
      netlist_node_list[out_ind].input_indices.append(netlist_node.id)

    if netlist_node.type == mnds.Type.MACRO_PIN:
      if netlist_node.ref_node_id is None:
        raise ValueError(f"Macro pin missing ref macro for node: "
                         f"{id_to_name_map[netlist_node.id]}.")

      if not netlist_node.output_indices:
        netlist_node_list[netlist_node.ref_node_id].input_indices.append(
            netlist_node.id)
      else:
        netlist_node_list[netlist_node.ref_node_id].output_indices.append(
            netlist_node.id)

  area = 0
  for netlist_node in netlist_node_list:
    node_name = id_to_name_map[netlist_node.id]
    if len(netlist_node.output_indices) >= _HIGH_FANOUT:
      logging.warning("%s driving %d outputs.", node_name,
                      len(netlist_node.output_indices))

    if netlist_node.type in {mnds.Type.STDCELL, mnds.Type.MACRO}:
      if netlist_node.dimension is None:
        raise ValueError(f"Width and/or height not defined for: {node_name}.")

      area += netlist_node.dimension.width * netlist_node.dimension.height

    if netlist_node.type == mnds.Type.MACRO_PIN:
      if netlist_node.offset is None:
        raise ValueError(f"Macro pin missing offset coords: {node_name}.")
    if not netlist_node.output_indices and not netlist_node.input_indices:
      logging.info("Unconnected node found: %s", node_name)
    elif (netlist_node.type == mnds.Type.MACRO and
          netlist_node.coord is not None):
      for cind in itertools.chain(netlist_node.input_indices,
                                  netlist_node.output_indices):
        place_macro_pin(netlist_node_list[cind], netlist_node)

  logging.info(
      "Total area of the macros and stdcells: %s. "
      "Number nodes: %d.", area, len(netlist_node_list))

  return mnds.MetaNetlist(
      node=netlist_node_list, canvas=generate_canvas(area), total_area=area)


def read_netlist(netlist_filepath: str) -> mnds.MetaNetlist:
  """Read netlist.pb.txt file.

  Args:
    netlist_filepath: netlist proto file path. It is expected in the
      tf.GraphDef() format. If a file is extremely large
      (larger than 2147483647 bytes) then we should break it up into smaller
      files, and pass them as comma separated list.

  Returns:
    Converted MetaNetlist.

  Raises:
    ValueError is the netlist_filepath is empty or just composed with comma.
  """
  netlist_filepath_list = netlist_filepath.split(",")
  netlist_filepath_list = [f for f in netlist_filepath_list if f]
  if not netlist_filepath_list:
    raise ValueError("Please ensure the netlist_filepath is not Empty.")

  meta_graph = tf.compat.v1.MetaGraphDef()

  for single_netlist_filepath in netlist_filepath_list:
    with open(single_netlist_filepath, "r") as f:
      tf_graph = text_format.Parse(f.read(), tf.compat.v1.GraphDef())
    meta_graph.graph_def.MergeFrom(tf_graph)
  return convert_tfgraph_to_meta_netlist(meta_graph)
