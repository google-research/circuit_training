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
"""Library to convert a plc into a Dreamplace PlaceDB.

Convention:
 - indices of macros, ports, and pins in plc is named "_index" and "_indices".
 - indices of nodes, pins, and nets in PlaceDB is named "_id" and "ids".
"""
import pickle

from absl import logging
from dreamplace import PlaceDB
import numpy as np

# Internal gfile dependencies


def blockage_area(plc):
  return np.sum([(b[2] - b[0]) * (b[3] - b[1]) for b in plc.get_blockages()])


def np_array_of_array(py_list_of_list, dtype):
  """converts a Python list of list into a Numpy array of array."""
  return np.array([np.array(l, dtype=dtype) for l in py_list_of_list],
                  dtype=object)


def convert_canvas(db, plc):
  """Convert canvas information in plc into PlaceDB.

  Args:
    db: The PlaceDB instance.
    plc: The PlacementCost instance.
  """
  # NOTE(hqzhu): don't accepet external settings for:
  # placement bounding box row_height and site width.
  db.xl = 0
  db.yl = 0
  db.xh, db.yh = plc.get_canvas_width_height()
  num_columns, num_rows = plc.get_grid_num_columns_rows()
  db.row_height = (db.yh / num_rows)
  db.site_width = (db.xh / num_columns)
  db.rows = []
  for i in range(num_rows):
    db.rows.append([0, i * db.row_height, db.xh, (i + 1) * db.row_height])


def convert_nodes(db, plc, hard_macro_order):
  """Convert nodes in plc into PlaceDB.

  Ports, hard macros, soft macros, and stdcells are converted to "nodes" in
  PlaceDB. Ports are considered "non-movable nodes". Soft macros and stdcells
  are considered "movable nodes". By default, we consider all the hard macros
  as "non-movable nodes".

  Node positions are different in two formats.
  Centered position is saved in the PlacementCost instance,
  while lower left position is saved in PlaceDB instance.

  Args:
    db: The PlaceDB instance.
    plc: The PlacementCost instance.
    hard_macro_order: Order of hard macros (excluding fixed ones).

  Returns:
    physical_node_indices: List of node indices in plc.
    node_index_to_node_id_map: Mapping from node index in plc to node id in
        PlaceDB.
    soft_macro_and_stdcell_indices: List of driver pin indices in plc.
    hard_macro_indices: List of hard macro indices.
    non_movable_node_indices: List of non-movable node indices in plc.
    num_blockage_dummy_node: Number of dummy blockages.
  """
  db.node_names = []
  db.node_name2id_map = {}
  db.node_x = []
  db.node_y = []
  db.node_orient = []
  # We keep a copy of the node_size_x and y, so when we restore them after
  # calling PlaceDB.__call__ which modifies them, in case if we want to call
  # the function again.
  db.original_node_size_x = []
  db.original_node_size_y = []
  db.macro_mask = []

  # To support, dreamplace mixed-size for crowded blocks, and to avoid
  # converting plc for every dreamplace call, we separate the non-fixed macros
  # and other movable nodes.
  soft_macro_and_stdcell_indices = []
  hard_macro_indices = hard_macro_order
  non_movable_node_indices = []

  for node_index in range(plc.num_nodes()):
    node_type = plc.get_node_type(node_index)
    if plc.is_node_fixed(node_index):
      assert node_index not in hard_macro_indices, (
          f'hard macro {node_index} is fixed and should '
          'not be included in `hard_macro_order`.')
      non_movable_node_indices.append(node_index)
    elif node_type == 'MACRO':
      if plc.is_node_soft_macro(node_index):
        soft_macro_and_stdcell_indices.append(node_index)
      else:
        # Ignoring the hard-macros, since they will be added based on the order
        # in `hard_macro_order`.
        pass
    elif node_type == 'STDCELL':
      soft_macro_and_stdcell_indices.append(node_index)
    elif node_type == 'PORT':
      non_movable_node_indices.append(node_index)
  # DREAMPlace requires nodes to be arragned movable-first, so do that.
  physical_node_indices = soft_macro_and_stdcell_indices + hard_macro_indices + non_movable_node_indices

  for node_id, node_index in enumerate(physical_node_indices):
    name = plc.get_node_name(node_index)
    db.node_names.append(name)
    db.node_name2id_map[name] = node_id
    if not plc.is_node_placed(node_index):
      logging.log_first_n(0, 'Node %s is not placed. Placed it at (0, 0).', 5,
                          node_index)
      x, y = (0, 0)
    else:
      x, y = plc.get_node_location(node_index)
      logging.log_first_n(0, 'Node %s is placed at (%f, %f).', 5, node_index, x,
                          y)
    if plc.get_node_type(node_index) == 'PORT':
      # Treat a port as a node with 0 dimension and 'N' orientation.
      db.node_orient.append(b'N')
      db.original_node_size_x.append(0)
      db.original_node_size_y.append(0)
    else:
      db.node_orient.append(plc.get_macro_orientation(node_index))
      w, h = plc.get_node_width_height(node_index)
      db.original_node_size_x.append(w)
      db.original_node_size_y.append(h)
    # DREAMPlace uses lower left position, while plc uses centered position.
    db.node_x.append(x - db.original_node_size_x[-1] / 2)
    db.node_y.append(y - db.original_node_size_y[-1] / 2)

  # if the blockage rate is 1, translate it into a dummy fixed node.
  num_blockage_dummy_node = 0
  for b in plc.get_blockages():
    # b is a tupe of (minx, miny, maxx, maxy, blockage_rate)
    if b[4] == 1:
      dummy_node_name = 'blockage_dummy_node_' + str(num_blockage_dummy_node)
      db.node_names.append(dummy_node_name)
      db.node_name2id_map[dummy_node_name] = len(
          physical_node_indices) + num_blockage_dummy_node
      db.node_x.append(b[0])
      db.node_y.append(b[1])
      db.original_node_size_x.append(b[2] - b[0])
      db.original_node_size_y.append(b[3] - b[1])
      db.node_orient.append(b'N')
      num_blockage_dummy_node += 1

  db.num_physical_nodes = len(physical_node_indices) + num_blockage_dummy_node

  db.num_terminals = (
      len(hard_macro_indices) + len(non_movable_node_indices) +
      num_blockage_dummy_node)
  db.macro_mask = [False] * len(soft_macro_and_stdcell_indices)

  db.node_size_x = db.original_node_size_x
  db.node_size_y = db.original_node_size_y
  db.num_non_movable_macros = len(hard_macro_indices)

  node_index_to_node_id_map = {
      n: i for i, n in enumerate(physical_node_indices)
  }

  return (physical_node_indices, node_index_to_node_id_map,
          soft_macro_and_stdcell_indices, hard_macro_indices,
          non_movable_node_indices, num_blockage_dummy_node)


def get_parent_node_index(plc, pin_index):
  """Returns the index of the parent node of pin_index."""
  node_type = plc.get_node_type(pin_index)
  if node_type == 'PORT' or node_type == 'STDCELL':
    # This is a virtual pin. The pin_index is the same as the index of the port
    # or the stdcell. We return the same index as the parent node index because
    # in this case, the port or the stdcell node itself should be considered as
    # the parent node of this virtual pin.
    return pin_index
  node_index = plc.get_ref_node_id(pin_index)
  if node_index == -1:
    return pin_index
  return node_index


def get_pin_offset(plc, node_index, pin_index):
  """Returns the x and y offsets from pin_index to node_index."""
  node_x, node_y = plc.get_node_location(node_index)
  if plc.get_node_type(node_index) == 'PORT':
    # Treat a port as a node with 0 dimension and 'N' orientation.
    w, h = 0, 0
  else:
    w, h = plc.get_node_width_height(node_index)
  pin_x, pin_y = plc.get_node_location(pin_index)
  return pin_x - (node_x - w / 2), pin_y - (node_y - h / 2)


def convert_a_net(db, plc, driver_pin_index, node_index_to_node_id_map,
                  pin_id_to_pin_index, counters):
  """Convert a single net in plc into PlaceDB.

  The net is driven by the virtual pin whose index is "driver_pin_index".

  Args:
    db: The PlaceDB instance.
    plc: The PlacementCost instance.
    driver_pin_index: List of driver pin indices in plc.
    node_index_to_node_id_map: Mapping from node index in plc to node id in
      PlaceDB.
    pin_id_to_pin_index: List of pin index in plc to pin id in PlaceDB.
    counters: Global counters to update.
  """
  net_id = counters['net_id']
  pin_id = counters['pin_id']

  # Use name of the driver pin as the name of the net.
  net_name = plc.get_node_name(driver_pin_index)
  db.net_names.append(net_name)
  db.net_name2id_map[net_name] = net_id
  # TODO(b/255357659): support get_node_weight.
  # NOTE(hqzhu): set net weight to deault 1.
  db.net_weights.append(1.0)
  # db.net_weights.append(plc.get_node_weight(driver_pin_index))
  # Add the driver pin to the list of pins.
  db.pin2net_map.append(net_id)
  db.pin_direct.append('OUTPUT')
  parent_node_index = get_parent_node_index(plc, driver_pin_index)
  offset_x, offset_y = get_pin_offset(plc, parent_node_index, driver_pin_index)
  db.pin_offset_x.append(offset_x)
  db.pin_offset_y.append(offset_y)
  parent_node_id = node_index_to_node_id_map[parent_node_index]
  db.node2pin_map[parent_node_id].append(pin_id)
  db.pin2node_map.append(parent_node_id)
  pin_ids_of_net = [pin_id]
  pin_id_to_pin_index.append(driver_pin_index)
  pin_id += 1

  for other_pin_index in plc.get_fan_outs_of_node(driver_pin_index):
    # Add the other pins to the list of pins.
    db.pin2net_map.append(net_id)
    db.pin_direct.append('INPUT')
    parent_node_index = get_parent_node_index(plc, other_pin_index)
    offset_x, offset_y = get_pin_offset(plc, parent_node_index, other_pin_index)
    db.pin_offset_x.append(offset_x)
    db.pin_offset_y.append(offset_y)
    parent_node_id = node_index_to_node_id_map[parent_node_index]
    db.node2pin_map[parent_node_id].append(pin_id)
    db.pin2node_map.append(parent_node_id)
    pin_ids_of_net.append(pin_id)
    pin_id_to_pin_index.append(other_pin_index)
    pin_id += 1
  db.net2pin_map.append(pin_ids_of_net)
  net_id += 1

  # Update counters.
  counters['net_id'] = net_id
  counters['pin_id'] = pin_id


def convert_pins_and_nets(db, plc, physical_node_indices,
                          node_index_to_node_id_map):
  """Convert pins and nets to PlaceDB.

  Here we work with the concerpt of a "virtual pin". Virtual pin is introduced
  to handle the differences when translating pins from plc to PlaceDB:
   - A net in PlaceDB only connect pins. In plc, a net also connects ports with
     pins.
   - A pin in PlaceDB can only belong to a single net. In plc, input pins of
     soft macros are collapsed into a single pin, which belongs to multiple
     nets.

  A virtual pin in plc is defined as one of:
   - An output pin of a macro.
   - An input pin of a hard macro.
   - An output pin of a soft macro, split into N virtual pins, one for each
     net the output pin belongs to. All N virtual pins have the same index as
     the output pin.
   - An imaginative pin sitting on a port. Its index is the same as the index of
     the incoming port, and its parent node is the port. Its offset is 0.

  Virtual pins satisfy the following properties:
   - Each virtual pin has a parent node.
   - Each virtual pin belongs to only a single net in the plc.
   - Nets in plc only connect virtual pins together.

  In this function, we will map each virtual pin in plc to a "pin" in PlaceDB,
  and each net in plc to a "net" in PlaceDB.

  Args:
    db: The PlaceDB instance.
    plc: The PlacementCost instance.
    physical_node_indices: List of node indices in plc.
    node_index_to_node_id_map: Mapping from node index in plc to node id in
      PlaceDB.

  Returns:
    driver_pin_indices: List of driver pin indices in plc.
    pin_id_to_pin_index: List of pin index in plc to pin id in PlaceDB.
  """
  db.pin_direct = []  # Array, len = number of pins
  db.pin_offset_x = []  # Array, len = number of pins
  db.pin_offset_y = []  # Array, len = number of pins
  db.node2pin_map = []  # Array of array. node id to pin ids
  for _ in range(db.num_physical_nodes):
    db.node2pin_map.append([])
  db.pin2node_map = []  # Array, len = number of pins
  db.net_name2id_map = {}
  db.net_names = []
  db.pin2net_map = []  # Array, len = number of pins
  db.net2pin_map = []
  db.net_weights = []

  counters = {'pin_id': 0, 'net_id': 0}
  driver_pin_indices = []
  pin_id_to_pin_index = []

  for node_index in physical_node_indices:
    node_type = plc.get_node_type(node_index)
    if node_type == 'PORT' or node_type == 'STDCELL':
      if plc.get_fan_outs_of_node(node_index):
        # This is a port or a stdcell that has fanouts. A virtual pin drives a
        # net from it.
        driver_pin_indices.append(node_index)
        driver_pin_index = node_index
        convert_a_net(db, plc, driver_pin_index, node_index_to_node_id_map,
                      pin_id_to_pin_index, counters)
    elif node_type == 'MACRO':
      # Output pins of both hard and soft macros drive pins.
      for driver_pin_index in plc.get_fan_outs_of_node(node_index):
        driver_pin_indices.append(driver_pin_index)
        convert_a_net(db, plc, driver_pin_index, node_index_to_node_id_map,
                      pin_id_to_pin_index, counters)

  return driver_pin_indices, pin_id_to_pin_index


def convert_to_ndarray(db):
  """Converts lists in the PlaceDB into Numpy arrays."""
  db.rows = np.array(db.rows, dtype=db.dtype)
  db.node_names = np.array(db.node_names, dtype=np.string_)
  db.node_x = np.array(db.node_x, dtype=db.dtype)
  db.node_y = np.array(db.node_y, dtype=db.dtype)
  db.node_orient = np.array(db.node_orient, dtype=np.string_)
  db.original_node_size_x = np.array(db.original_node_size_x, dtype=db.dtype)
  db.original_node_size_y = np.array(db.original_node_size_y, dtype=db.dtype)
  db.node_size_x = np.array(db.node_size_x, dtype=db.dtype)
  db.node_size_y = np.array(db.node_size_y, dtype=db.dtype)
  db.node2pin_map = np_array_of_array(db.node2pin_map, dtype=np.int32)
  db.pin_direct = np.array(db.pin_direct, dtype=np.string_)
  db.pin_offset_x = np.array(db.pin_offset_x, dtype=db.dtype)
  db.pin_offset_y = np.array(db.pin_offset_y, dtype=db.dtype)
  db.pin2node_map = np.array(db.pin2node_map, dtype=np.int32)
  db.pin2net_map = np.array(db.pin2net_map, dtype=np.int32)
  db.net_names = np.array(db.net_names, dtype=np.string_)
  db.net_weights = np.array(db.net_weights, dtype=db.dtype)
  db.net2pin_map = np_array_of_array(db.net2pin_map, dtype=np.int32)
  db.flat_node2pin_map, db.flat_node2pin_start_map = db.flatten_nested_map(
      db.pin2node_map, db.node2pin_map)
  db.flat_net2pin_map, db.flat_net2pin_start_map = db.flatten_nested_map(
      db.pin2net_map, db.net2pin_map)
  db.macro_mask = np.array(db.macro_mask, dtype=np.uint8)


def initialize_placedb_region_attributes(db):
  """Initialize the region-related attributes in PlaceDB instance.

  Args:
    db: The PlaceDB instance.  Assume there is no region constraints in the plc
      format.
  """
  db.regions = []
  db.flat_region_boxes = np.array([], dtype=db.dtype)
  db.flat_region_boxes_start = np.array([0], dtype=np.int32)
  db.node2fence_region_map = np.array([], dtype=np.int32)


class PlcConverter(object):
  """Class that converts a plc into a Dreamplace PlaceDB."""

  def __init__(self):
    # List of movable node (except hard macros) indices in plc.
    self._soft_macro_and_stdcell_indices = None
    # List of hard macros in plc.
    self._hard_macro_indices = None
    # List of non-movable node indices in plc.
    self._non_movable_node_indices = None
    # List of driver pin indices in plc.
    self._driver_pin_indices = None
    # Mapping from node index in plc to node id in PlaceDB.
    self._node_index_to_node_id_map = None
    # List of pin index in plc to pin id in PlaceDB.
    self._pin_id_to_pin_index = None

  @property
  def soft_macro_and_stdcell_indices(self):
    return self._soft_macro_and_stdcell_indices

  @property
  def hard_macro_indices(self):
    return self._hard_macro_indices

  @property
  def non_movable_node_indices(self):
    return self._non_movable_node_indices

  @property
  def driver_pin_indices(self):
    return self._driver_pin_indices

  @property
  def node_index_to_node_id_map(self):
    return self._node_index_to_node_id_map

  @property
  def pin_id_to_pin_index(self):
    return self._pin_id_to_pin_index

  def non_movable_macro_area(self, plc, num_non_movable_macros=None):
    """Returns the area of the non-movable macros.

    Args:
      plc: A PlacementCost object.
      num_non_movable_macros: Optional int. spesifies the number of placed
        macros that should be consider as non-movable.
    """
    if not num_non_movable_macros:
      num_non_movable_macros = len(self._hard_macro_indices)

    non_movable_macro = self._hard_macro_indices[:num_non_movable_macros]

    macro_width_height = [
        plc.get_node_width_height(m) for m in non_movable_macro
    ]
    macro_area = np.sum([w * h for w, h in macro_width_height])

    return macro_area

  def convert(self, plc, hard_macro_order=None):
    """Converts a plc into a Dreamplace PlaceDB format.

    Args:
      plc: The PlacementCost instance.
      hard_macro_order: Optional list of macros ordered by how the RL agent will
        place them. If not provided, use the node order of the plc.

    Returns:
      The converted PlaceDB instance.
    """
    db = PlaceDB.PlaceDB()
    db.dtype = np.float32

    if not hard_macro_order:
      hard_macro_order = [
          m for m in plc.get_macro_indices()
          if not (plc.is_node_soft_macro(m) or plc.is_node_fixed(m))
      ]

    convert_canvas(db, plc)
    (physical_node_indices, self._node_index_to_node_id_map,
     self._soft_macro_and_stdcell_indices, self._hard_macro_indices,
     self._non_movable_node_indices,
     self._num_blockage_dummy_node) = convert_nodes(db, plc, hard_macro_order)
    self._driver_pin_indices, self._pin_id_to_pin_index = convert_pins_and_nets(
        db, plc, physical_node_indices, self._node_index_to_node_id_map)

    db.total_space_area = db.xh * db.yh - blockage_area(
        plc) - self.non_movable_macro_area(plc)
    convert_to_ndarray(db)
    initialize_placedb_region_attributes(db)
    return db

  def convert_and_dump(self, plc, path_to_placedb, hard_macro_order=None):
    """Converts a plc into a dreamplace.PlaceDB format and dump it.

    Args:
      plc: The PlacementCost instance.
      path_to_placedb: the path to the output file.
      hard_macro_order: Optional list of macros ordered by how the RL agent will
        place them. If not provided, use the node order of the plc.

    Returns:
      The converted PlaceDB instance.
    """
    db = self.convert(plc, hard_macro_order)
    with open(path_to_placedb, 'wb') as output_file:
      pickle.dump(db, output_file)
    return db

  def update_macro(self, db, plc, macro_index):
    """Updates information about a macro from plc into db."""
    if not plc.is_node_placed(macro_index):
      return

    is_soft_macro = plc.is_node_soft_macro(macro_index)
    # Update macro location.
    macro_id = self._node_index_to_node_id_map[macro_index]
    x, y = plc.get_node_location(macro_index)
    w, h = plc.get_node_width_height(macro_index)
    db.node_x[macro_id], db.node_y[macro_id] = x - w / 2, y - h / 2
    # Update macro orientation.
    old_orient = db.node_orient[macro_id]
    new_orient = plc.get_macro_orientation(macro_index)
    db.node_orient[macro_id] = new_orient
    # Update offsets of hard macro pins only when orientation has changed.
    if old_orient != new_orient and not is_soft_macro:
      for pin_id in db.node2pin_map[macro_id]:
        pin_index = self._pin_id_to_pin_index[pin_id]
        pin_x, pin_y = get_pin_offset(plc, macro_index, pin_index)
        db.pin_offset_x[pin_id] = pin_x
        db.pin_offset_y[pin_id] = pin_y

  def update_num_non_movable_macros(self, db, plc, num_non_movable_macros):
    """Updates PlaceDB parameters give the new num_non_movable_macros."""
    db.num_terminals = num_non_movable_macros + len(
        self._non_movable_node_indices) + self._num_blockage_dummy_node
    macro_mask = [False] * len(
        self._soft_macro_and_stdcell_indices) + [True] * (
            len(self._hard_macro_indices) - num_non_movable_macros)
    db.macro_mask = np.array(macro_mask, dtype=np.uint8)
    db.total_space_area = db.xh * db.yh - blockage_area(
        plc) - self.non_movable_macro_area(plc, num_non_movable_macros)
    # These parameters are changed after calling to PlaceDB.__call__,
    # resetting them to their original values.
    db.node_size_x = db.original_node_size_x
    db.node_size_y = db.original_node_size_y
    db.num_movable_pins = None
    db.num_non_movable_macros = num_non_movable_macros
