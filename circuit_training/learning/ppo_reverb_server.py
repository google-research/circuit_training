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
"""Main binary to launch a stand alone Reverb RB server."""

import os

from absl import app
from absl import flags

from circuit_training.learning import ppo_reverb_server_lib

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer(
    'replay_buffer_capacity', 1024,
    'Capacity of the replay buffer table. Please set this to '
    'larger than num_episodes_per_iteration.')
flags.DEFINE_integer('port', None, 'Port to start the server on.')
flags.DEFINE_integer(
    'global_seed', 111,
    'Used in env and weight initialization, does not impact action sampling.')

FLAGS = flags.FLAGS


def main(_):
  # Create the path for the serialized collect policy.
  root_dir = os.path.join(FLAGS.root_dir, str(FLAGS.global_seed))
  ppo_reverb_server_lib.start_reverb_server(root_dir,
                                            FLAGS.replay_buffer_capacity,
                                            FLAGS.port)


if __name__ == '__main__':
  flags.mark_flags_as_required(['root_dir', 'port'])
  app.run(main)
