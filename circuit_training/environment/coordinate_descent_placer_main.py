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

The algorithm runs for a given number of epochs (iterations).
For each iteartion, for each node by a given --cd_node_order, place the node
greedily on the best grid location.
If --cd_use_stdcell_placer is True, place hard macros greedily first,
then followed by stdcell placer to place all stdcells.

When --cd_epochs=1, this algorithm is equivalent to greedy algorithm.

Example usage:

python circuit_training/environment/coordinate_descent_placer_main.py
--netlist_file "/path/to/netlist.pb.txt"
--init_placement "/path/to/initial_placement.plc"
"""

import functools

from absl import app
from absl import flags
from circuit_training.environment import coordinate_descent_placer
from circuit_training.environment import environment
from circuit_training.environment import placement_util
import numpy as np

flags.DEFINE_string('netlist_file', None, 'Path to netlist file.')
flags.DEFINE_string('init_placement', None, 'Path to initial placement file.')
flags.DEFINE_string('cd_output_dir', '/tmp/cd', 'CD output dir.')
flags.DEFINE_string('cd_placement_filename', 'cd', 'CD placement filename.')

FLAGS = flags.FLAGS


def main(_):
  np.random.seed(FLAGS.seed)

  plc = placement_util.create_placement_cost(FLAGS.netlist_file,
                                             FLAGS.init_placement)

  if not FLAGS.cd_use_init_location:
    plc.unplace_all_nodes()

  def cost_fn(plc):
    return environment.cost_info_function(plc=plc, done=True)

  cost_fn = functools.partial(
      cost_fn, wirelength_weight=1.0, density_weight=0.1, congestion_weight=0.1)

  placer = coordinate_descent_placer.CoordinateDescentPlacer(plc, cost_fn)

  placer.place()
  placer.save_placement(FLAGS.cd_output_dir,
                        f'{FLAGS.cd_placement_filename}.plc')
  print(f'Final CD placement can be found at {FLAGS.cd_output_dir}')


if __name__ == '__main__':
  app.run(main)
