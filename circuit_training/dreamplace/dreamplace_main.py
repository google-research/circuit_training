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
"""A binary place stdcells of a Morpheus netlist using Dreamplace."""

import random

from absl import app
from absl import flags
from circuit_training.dreamplace import dreamplace_core
from circuit_training.dreamplace import dreamplace_util
from dreamplace import Params
import numpy as np
import torch

_SEED = flags.DEFINE_integer(
    'seed', 0, 'RNG seed for all algorithms.'
    'If None, RNG is not seeded.')
_RUN_FD_AS_BASELINE = flags.DEFINE_bool(
    'run_fd_as_baseline', False,
    'If True, run FD and report proxy cost and run time, as a baseline.')
_DP_MAKE_SOFT_MACROS_SQUARE_FLAG = flags.DEFINE_bool(
    'dp_make_soft_macros_square_flag', True,
    'If true, make soft macros square for dreamplace.')
_DP_HARD_MACRO_MOVABLE = flags.DEFINE_bool(
    'dp_hard_macro_movable', False,
    'If true, hard macros are allowed to move in dreamplace.')
_DP_GPU = flags.DEFINE_bool(
    'dp_gpu', False,
    'If true, run dreamplace on GPU. Otherwise, run dreamplace on CPU.')
_DP_LEGALIZATION = flags.DEFINE_bool(
    'dp_legalization', False,
    'If true, run legalization after global placement in dreamplace.')
# DREAMPlace parameters.
_DP_ITERATION = flags.DEFINE_integer(
    'dp_iteration', 1000,
    'Dreamplace parameter. Maximum number of iterations to optimized.')
_DP_TARGET_DENSITY = flags.DEFINE_float(
    'dp_target_density', 1.0, 'Target density parameter to set in DREAMPlace.')
_DP_LEARNING_RATE = flags.DEFINE_float(
    'dp_learning_rate', '0.01',
    'Dreamplace parameter. Learning rate of optimizer.')
_DP_STOP_OVERFLOW = flags.DEFINE_float(
    'dp_stop_overflow', 0.1,
    'Stop oveflow as the stopping criteria of dreamplace.')
_DP_NUM_BINS_X = flags.DEFINE_integer(
    'dp_num_bins_x', None,
    'Dreamplace parameter. Optional number of bins in horizontal direction.')
_DP_NUM_BINS_Y = flags.DEFINE_integer(
    'dp_num_bins_y', None,
    'Dreamplace parameter. Optional number of bins in vertical direction.')
# Inputs and outputs.
_NETLIST_FILE = flags.DEFINE_string('netlist_file', None,
                                    'Path to the input netlist file.')
_INIT_PLACEMENT = flags.DEFINE_string(
    'init_placement', None, 'Path to file containing initial placement of '
    'nodes. Used to initialize layout of nodes for replacer environment type '
    'or sa placer.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None,
                                  'Base directory to output logs and results.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)
  # NOTE(hqzhu): pass args of netlist info to load_plc instead of using flags.
  plc = dreamplace_util.load_plc(
      netlist_file=_NETLIST_FILE.value,
      make_soft_macros_square_flag=_DP_MAKE_SOFT_MACROS_SQUARE_FLAG.value,
      output_dir=_OUTPUT_DIR.value,
      init_placement=_INIT_PLACEMENT.value)
  canvas_width, canvas_height = plc.get_canvas_width_height()
  dp_params = Params.get_dreamplace_params(
      _DP_ITERATION.value, _DP_TARGET_DENSITY.value, _DP_LEARNING_RATE.value,
      canvas_width, canvas_height, _DP_NUM_BINS_X.value, _DP_NUM_BINS_Y.value,
      _DP_GPU.value, _OUTPUT_DIR.value, _DP_LEGALIZATION.value,
      _DP_STOP_OVERFLOW.value)
  if _DP_GPU.value:
    torch.cuda.init()
  dreamplace_core.optimize_using_dreamplace(plc, dp_params, _OUTPUT_DIR.value,
                                            _DP_HARD_MACRO_MOVABLE.value)


if __name__ == '__main__':
  app.run(main)
