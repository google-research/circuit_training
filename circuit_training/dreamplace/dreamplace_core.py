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
"""A soft macro placer using Dreamplace."""
import time
from absl import logging
from circuit_training.dreamplace import dreamplace_util
from circuit_training.dreamplace import placedb_plc
from dreamplace import NonLinearPlace


class SoftMacroPlacer(object):
  """A soft macro placer using Dreamplace."""

  def __init__(self, plc, params, hard_macro_order=None):
    self.params = params
    self.placedb_plc = placedb_plc.PlacedbPlc(plc, params, hard_macro_order)

  # NOTE(hqzhu): convergence flag may not be right.
  # We cannot simply check divergence based on #iterations in DP V3.
  def place(self) -> bool:
    """Place soft macros.

    Returns:
      a bool indicating if DP converges or not based on checking #iterations.
    """
    nonlinear_place = NonLinearPlace.NonLinearPlace(self.params,
                                                    self.placedb_plc.placedb)
    metrics = nonlinear_place(self.params, self.placedb_plc.placedb)
    logging.info('Last Dreamplace metric: %s', str(metrics[-1][-1][-1]))
    total_iterations = sum(
        [stage['iteration'] for stage in self.params.global_place_stages])

    return (metrics[-1][-1][-1].iteration) < total_iterations


def optimize_using_dreamplace(plc,
                              params,
                              output_dir,
                              hard_macro_movable=False):
  """Optimzes using Dreamplace."""
  # Initialization, slow but only happens once.
  start_init_time = time.time()
  placer = SoftMacroPlacer(plc, params)
  if hard_macro_movable:
    placer.placedb_plc.update_num_non_movable_macros(
        plc, num_non_movable_macros=0)
  logging.info('Initializing Dreamplace took %g seconds.',
               time.time() - start_init_time)

  # Dreamplace optimzation.
  start_opt_time = time.time()
  placer.place()
  logging.info('Dreamplace optimization took %g seconds.',
               time.time() - start_opt_time)

  # Write the optimized stdcell location back to the plc. This step may be
  # omitted if the Dreamplace reported density can be used in our cost function
  # directly.
  start_write_time = time.time()
  placer.placedb_plc.write_movable_locations_to_plc(plc)
  logging.info('Writing soft macro locations to plc took %g seconds.',
               time.time() - start_write_time)

  # The total run time of using Dreamplace to optimize soft macro placement.
  duration = time.time() - start_opt_time
  filename_prefix = 'dreamplace'
  dreamplace_util.print_and_save_result(plc, duration, 'Dreamplace',
                                        filename_prefix, output_dir)
