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
"""Util functions for modify meta_netlist."""
from circuit_training.grouping import meta_netlist_data_structure as mnds


def set_canvas_width_height(meta_netlist: mnds.MetaNetlist, canvas_width: float,
                            canvas_height: float) -> None:
  """Sets canvas width and height."""
  meta_netlist.canvas.dimension.width = canvas_width
  meta_netlist.canvas.dimension.height = canvas_height


def set_canvas_columns_rows(meta_netlist: mnds.MetaNetlist, canvas_columns: int,
                            canvas_rows: int) -> None:
  """Sets canvas columns and rows."""
  meta_netlist.canvas.num_columns = canvas_columns
  meta_netlist.canvas.num_rows = canvas_rows


def disconnect_single_net(meta_netlist: mnds.MetaNetlist,
                          node_index: int) -> None:
  """Disconnects a single net."""
  node = meta_netlist.node[node_index]
  for out_index in node.output_indices:
    meta_netlist.node[out_index].input_indices.remove(node_index)
  node.output_indices = []


def disconnect_high_fanout_nets(meta_netlist: mnds.MetaNetlist,
                                max_allowed_fanouts: int = 500) -> None:
  """Disconnect all the nodes whose output_indices exceeds max_allowed_fanouts.

  Args:
    meta_netlist: Meta Netlist.
    max_allowed_fanouts: Maximum allowed fanouts.
  """
  for index, node in enumerate(meta_netlist.node):
    if (node.type in {mnds.Type.PORT, mnds.Type.STDCELL, mnds.Type.MACRO_PIN}
        and len(node.output_indices) > max_allowed_fanouts):
      disconnect_single_net(meta_netlist, index)
