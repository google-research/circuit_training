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

import math
import os
import time
from typing import Optional

from absl import logging
import circuit_training.environment.placement_util as util
from dreamplace import Params
import gin


def print_and_save_result(
    plc, duration, method_name, filename_prefix, output_dir
):
  """Prints out proxy costs and run time. Writes out plc and svg."""
  plc_filename = os.path.join(output_dir, f'{filename_prefix}.plc')
  util.save_placement(plc, plc_filename)
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


def load_plc(netlist_file, output_dir, init_placement=None):
  """Loads the netlist and initial plc."""
  t = time.time()
  plc = util.create_placement_cost(
      netlist_file=netlist_file, init_placement=init_placement
  )
  duration = time.time() - t
  print_and_save_result(
      plc, duration, 'Loading initial PLC', 'dreamplace_initial', output_dir
  )
  return plc


def get_bin_size(
    width: float, height: float, num_bins: int = 128 * 128
) -> tuple[int, int]:
  """Find the power two bin size closest to the canvas ratio."""
  num_bins_x = math.sqrt(num_bins * width / height)
  num_bins_x = int(math.pow(2, round(math.log(num_bins_x) / math.log(2))))
  # num_bins_x should be between 1 and num_bins.
  num_bins_x = max(min(num_bins_x, num_bins), 1)
  num_bins_y = int(num_bins / num_bins_x)
  return num_bins_x, num_bins_y


@gin.configurable
def get_dreamplace_params(
    iteration: int = 1000,
    # The target density should be customized for each project. We set it to a
    # pessimistic value because it gives us the most similar std cell placements
    # compared to PNR placements of EDA tools.
    target_density: float = 0.425,
    learning_rate: float = 0.01,
    canvas_width: Optional[float] = None,
    canvas_height: Optional[float] = None,
    num_bins_x: Optional[int] = None,
    num_bins_y: Optional[int] = None,
    gpu: bool = False,
    result_dir: str = 'results',
    legalize_flag: bool = False,
    stop_overflow: float = 0.1,
    routability_opt_flag: bool = False,
    num_threads: int = 8,
    deterministic_flag: bool = False,
    regioning: bool = False,
):
  """Returns the parameters to config Dreamplace."""
  params = Params.Params()
  # enable dreamplace for Morpheus run.
  params.circuit_training_mode = True

  if num_bins_x and num_bins_y:
    params.num_bins_x = num_bins_x
    params.num_bins_y = num_bins_y
  elif canvas_width and canvas_height:
    params.num_bins_x, params.num_bins_y = get_bin_size(
        canvas_width, canvas_height
    )
    logging.info(
        'Update num_bins_x and num_bins_y: (%s, %s)',
        params.num_bins_x,
        params.num_bins_y,
    )
  else:
    params.num_bins_x = 128
    params.num_bins_y = 128
    logging.warning(
        'Niether bin size or canvas size is provided, '
        'use the default num_bins: 128x128.'
    )
  params.global_place_stages = [{
      'num_bins_x': params.num_bins_x,
      'num_bins_y': params.num_bins_y,
      'iteration': iteration,
      'learning_rate': learning_rate,
      'wirelength': 'weighted_average',
      'optimizer': 'nesterov',
  }]

  params.legalize_flag = legalize_flag
  # disable detailed placement
  params.detailed_place_flag = False
  params.target_density = target_density
  params.density_weight = 8e-5
  params.gpu = gpu
  params.result_dir = result_dir
  params.stop_overflow = stop_overflow
  params.regioning = regioning

  # routability optimization flags:
  params.routability_opt_flag = routability_opt_flag
  # disable nctugr for routing congestion optimization
  params.adjust_nctugr_area_flag = False

  # The number of threads.
  params.num_threads = num_threads

  params.deterministic_flag = deterministic_flag
  if deterministic_flag:
    params.enable_fillers = False
    params.gp_noise_ratio = 0.001
    params.random_center_init_flag = False

  return params
