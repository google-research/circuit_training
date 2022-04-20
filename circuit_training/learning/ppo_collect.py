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
"""Collection job using a variable container for policy updates."""

import functools
import os

from absl import app
from absl import flags

from circuit_training.environment import environment
from circuit_training.learning import ppo_collect_lib

from tf_agents.system import system_multiprocessing as multiprocessing


FLAGS = flags.FLAGS

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
flags.DEFINE_integer(
    'task_id', 0, 'Identifier of the collect task. Must be unique in a job.')
flags.DEFINE_integer(
    'write_summaries_task_threshold', 1,
    'Collect jobs with tas ID smaller than this value writes '
    'summaries only.')
flags.DEFINE_integer(
    'max_sequence_length', 134,
    'The sequence length for Reverb replay buffer. Depends on the environment.')
flags.DEFINE_integer(
    'global_seed', 111,
    'Used in env and weight initialization, does not impact action sampling.')


def main(_):
  root_dir = os.path.join(FLAGS.root_dir, str(FLAGS.global_seed))

  create_env_fn = functools.partial(
      environment.create_circuit_environment,
      netlist_file=FLAGS.netlist_file,
      init_placement=FLAGS.init_placement,
      global_seed=FLAGS.global_seed,
  )

  ppo_collect_lib.collect(
      task=FLAGS.task_id,
      root_dir=root_dir,
      replay_buffer_server_address=FLAGS.replay_buffer_server_address,
      variable_container_server_address=FLAGS.variable_container_server_address,
      create_env_fn=create_env_fn,
      max_sequence_length=FLAGS.max_sequence_length,
      write_summaries_task_threshold=FLAGS.write_summaries_task_threshold,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir', 'replay_buffer_server_address',
      'variable_container_server_address'
  ])
  multiprocessing.handle_main(functools.partial(app.run, main))
