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
"""Utility functions for dreamplace."""

import os
import time
from absl import logging
import circuit_training.environment.placement_util as util


def print_and_save_result(plc, duration, method_name, filename_prefix,
                          output_dir):
  """Prints out proxy costs and run time. Writes out plc and svg."""
  plc_filename = os.path.join(output_dir, f'{filename_prefix}.plc')
  util.save_placement(plc, plc_filename)
  # NOTE(hqzhu): remove save plc as svg feature.
  # since placement_util is not supported.
  logging.info('**************************************************')
  logging.info('*** %s took %g seconds.', method_name, duration)
  logging.info('***')
  logging.info('*** Wirelength: %g', plc.get_wirelength())
  logging.info('*** Wirelength cost: %g', plc.get_cost())
  logging.info('*** Density cost: %g', plc.get_density_cost())
  logging.info('*** Congestion cost: %g', plc.get_congestion_cost())
  logging.info('***')
  logging.info('*** Plc file: %s', plc_filename)
  logging.info('**************************************************\n')


def load_plc(netlist_file,
             make_soft_macros_square_flag,
             output_dir,
             init_placement=None):
  """Loads the netlist and initial plc."""
  t = time.time()
  plc = util.create_placement_cost(
      netlist_file=netlist_file, init_placement=init_placement)
  if make_soft_macros_square_flag:
    plc.make_soft_macros_square()
  duration = time.time() - t
  print_and_save_result(plc, duration, 'Loading initial PLC',
                        'dreamplace_initial', output_dir)
  return plc
