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
"""Sample training with distributed collection using a variable container."""

import functools
import os
import random

from absl import app
from absl import flags
from absl import logging

from circuit_training.environment import environment
from circuit_training.learning import train_ppo_lib


import numpy as np
import tensorflow as tf

from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import strategy_utils

flags.DEFINE_string('netlist_file', '', 'File path to the netlist file.')
flags.DEFINE_string('init_placement', '',
                    'File path to the init placement file.')
# TODO(b/219085316): Open source dreamplace.
flags.DEFINE_string(
    'std_cell_placer_mode', 'fd',
    'Options for fast std cells placement: `fd` (uses the '
    'force-directed algorithm), `dreamplace` (uses DREAMPlace '
    'algorithm).')
flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('replay_buffer_server_address', None,
                    'Replay buffer server address.')
flags.DEFINE_string('variable_container_server_address', None,
                    'Variable container server address.')
flags.DEFINE_integer('num_iterations', 10000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer(
    'sequence_length', 134,
    'The sequence length to estimate shuffle size. Depends on the environment.'
    'Max horizon = T translates to sequence_length T+1 because of the '
    'additional boundary step (last -> first).')
flags.DEFINE_integer(
    'num_episodes_per_iteration', 1024,
    'This is the number of episodes we train on in each iteration.')
flags.DEFINE_integer('global_batch_size', 1024,
                     'Global batch size across all replicas.')

flags.DEFINE_integer(
    'global_seed', 111,
    'Used in env and weight initialization, does not impact action sampling.')

FLAGS = flags.FLAGS


def main(_):

  logging.info('global seed=%d', FLAGS.global_seed)
  np.random.seed(FLAGS.global_seed)
  random.seed(FLAGS.global_seed)
  tf.random.set_seed(FLAGS.global_seed)

  root_dir = os.path.join(FLAGS.root_dir, str(FLAGS.global_seed))

  strategy = strategy_utils.get_strategy(FLAGS.tpu, FLAGS.use_gpu)

  create_env_fn = functools.partial(
      environment.create_circuit_environment,
      netlist_file=FLAGS.netlist_file,
      init_placement=FLAGS.init_placement,
      global_seed=FLAGS.global_seed)

  use_model_tpu = bool(FLAGS.tpu)

  batch_size = int(FLAGS.global_batch_size / strategy.num_replicas_in_sync)
  logging.info('global batch_size=%d', FLAGS.global_batch_size)
  logging.info('per-replica batch_size=%d', batch_size)

  train_ppo_lib.train(
      root_dir=root_dir,
      strategy=strategy,
      replay_buffer_server_address=FLAGS.replay_buffer_server_address,
      variable_container_server_address=FLAGS.variable_container_server_address,
      create_env_fn=create_env_fn,
      sequence_length=FLAGS.sequence_length,
      per_replica_batch_size=batch_size,
      num_iterations=FLAGS.num_iterations,
      num_episodes_per_iteration=FLAGS.num_episodes_per_iteration,
      use_model_tpu=use_model_tpu,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'replay_buffer_server_address',
      'variable_container_server_address',
  ])
  multiprocessing.handle_main(functools.partial(app.run, main))
