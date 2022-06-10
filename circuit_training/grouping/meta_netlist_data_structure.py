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
"""Data structure definition for meta netlist."""
import dataclasses
import enum
from typing import List, Optional


@enum.unique
class Type(enum.Enum):
  """Type enum."""
  UNSET = 0
  STDCELL = 1
  MACRO = 2
  PORT = 3
  MACRO_PIN = 4


@enum.unique
class Side(enum.IntEnum):
  """Side of the node."""
  TOP = 0
  RIGHT = 1
  BOTTOM = 2
  LEFT = 3


@dataclasses.dataclass
class CoordIndex:
  """Coord index. It is used in storing the grouping information."""
  coord: float
  id: int  # Node index.


@dataclasses.dataclass
class Coord:
  """Coord dataclass."""
  x: float
  y: float


@dataclasses.dataclass
class Dimension:
  """Dimension dataclass."""
  width: float
  height: float


@dataclasses.dataclass
class Offset:
  """Offset dataclass."""
  x: float
  y: float


class Orientation(enum.Enum):
  """Orientation enum.

  Orientation of a macro can be the following:
   ___     ___     ___     ___
  |  *|   |*  |   |   |   |   |
  |   |   |   |   |   |   |   |
  |   |   |   |   |*  |   |  *|
   ---     ---     ---     ---
  NORMAL, FLIP_X, FLIP_XY, FLIP_Y
  N,      FN,     S,       FS
  R0,     MY,     R180,    MX

   ______    ______   ______    ______
  |      |  |      | |*     |  |     *|
  |     *|  |*     | |      |  |      |
   ------    ------   ------    ------
  E,        FE,      W,        FW
  R270,     MX90,    R90,      MY90

  """
  N = 0
  R0 = 0
  NORMAL = 0

  FN = 1
  MY = 1
  FLIP_X = 1

  S = 2
  R180 = 2
  FLIP_XY = 2

  FS = 3
  MX = 3
  FLIP_Y = 3

  # Anything below here is a 90/270 degree rotation.
  # Do not renumber them, since there are checks to evaluate
  # width/height of a macro based on whether the rotation
  # is 90/270 degrees.
  E = 4
  R270 = 4

  FE = 5
  MX90 = 5
  MYR90 = 5

  W = 6
  R90 = 6

  FW = 7
  MY90 = 7
  MXR90 = 7


@dataclasses.dataclass
class BoundingBox:
  """Bounding Box dataclass."""
  minx: float
  maxx: float
  miny: float
  maxy: float


@dataclasses.dataclass
class Constraint:
  """Constraint dataclass."""
  side: Optional[Side] = None


@dataclasses.dataclass
class NetlistNode:
  """NetlistNode dataclass.

  A netlist node can be of type STDCELL, MACRO, PORT, or MACRO_PIN.
  It has the location, size, inputs and outputs information.
  """
  id: Optional[int] = 0
  name: Optional[str] = ""
  type: Optional[Type] = Type.UNSET

  weight: Optional[float] = None

  # Dimension is required for stdcells and macros.
  dimension: Optional[Dimension] = None

  # Used for macros.
  orientation: Optional[Orientation] = None

  # coord is x, y position. Refers to to center point for stdcells and macros.
  coord: Optional[Coord] = None

  # offset is used for macro_pins only. Note that, coord must be equal to
  # offset + ref_node_id's coord.
  offset: Optional[Offset] = None

  # ref_node_id is used for macro_pins. Refers to the macro it belongs to.
  ref_node_id: Optional[int] = None

  # We only have constraint for ports for now, can be expanded.
  constraint: Optional[Constraint] = None

  # These are set after the full netlist is read.
  # References to other NetlistNode id's.
  output_indices: Optional[List[int]] = dataclasses.field(default_factory=list)
  input_indices: Optional[List[int]] = dataclasses.field(default_factory=list)

  # If this node is a macro, use this boolean to check if it's a soft (made up
  # of standard cells), or hard (like an SRAM).
  soft_macro: Optional[bool] = False


@dataclasses.dataclass
class Canvas:
  """Canvas dataclass.

  Canvas is the placement area. It also has the information of how many rows
  and columns it's divided for placement use.
  """
  dimension: Dimension
  num_rows: int
  num_columns: int


@dataclasses.dataclass
class MetaNetlist:
  """MetaNetlist dataclass.

  MetaNetlist contains a node graph, a canvas and other meta informations.
  """
  node: List[NetlistNode] = dataclasses.field(default_factory=list)
  canvas: Optional[Canvas] = None

  # total_area refers to the sum of areas of all nodes.
  total_area: Optional[float] = 0.0
