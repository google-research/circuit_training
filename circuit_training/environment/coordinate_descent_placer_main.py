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
r"""A placer that implements coordinate descent algorithm.

The placer can start from a scratch (i.e., empty grid), or from an existing node
locations specified by --init_placement.

Example usage:

python circuit_training/environment/coordinate_descent_placer_main.py
--netlist_file "/path/to/netlist.pb.txt"
--init_placement "/path/to/initial_placement.plc"
"""

import os

from absl import app
from absl import flags
from circuit_training.environment import coordinate_descent_placer
from circuit_training.environment import environment
from circuit_training.environment import placement_util
import numpy as np

_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_NETLIST_FILE = flags.DEFINE_string(
    'netlist_file', None, 'Path to netlist file.'
)
_INIT_PLACEMENT = flags.DEFINE_string(
    'init_placement', None, 'Path to initial placement file.'
)
_CD_OUTPUT_DIR = flags.DEFINE_string(
    'cd_output_dir', '/tmp/cd', 'CD output dir.'
)
_CD_PLACEMENT_FILENAME = flags.DEFINE_string(
    'cd_placement_filename', 'cd_placement.plc', 'CD placement filename.'
)


def main(_):
  np.random.seed(_SEED.value)

  plc = placement_util.create_placement_cost(
      _NETLIST_FILE.value, _INIT_PLACEMENT.value
  )

  def cost_fn(plc):
    return environment.cost_info_function(plc=plc, done=True)

  placer = coordinate_descent_placer.CoordinateDescentPlacer(plc, cost_fn)

  placer.place()
  output_plc_file = os.path.join(
      _CD_OUTPUT_DIR.value, _CD_PLACEMENT_FILENAME.value
  )
  placement_util.save_placement(plc, output_plc_file)
  print(f'Final CD placement can be found at {output_plc_file}')


if __name__ == '__main__':
  app.run(main)
