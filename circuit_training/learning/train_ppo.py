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
