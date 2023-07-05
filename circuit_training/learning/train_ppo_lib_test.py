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
"""Tests for train_ppo_lib."""

from circuit_training.learning import train_ppo_lib
from tf_agents.utils import test_utils


class TrainPpoLibTest(test_utils.TestCase):

  def test_compute_init_iteration(self):
    init_train_step = 100
    # The following parameters are divisors.
    sequence_length = 10
    num_episodes_per_iteration = 1
    num_epochs = 2
    # The following parameters are mulipliers.
    per_replica_batch_size = 2
    num_replicas_in_sync = 3

    init_iteration = train_ppo_lib.compute_init_iteration(
        init_train_step=init_train_step,
        sequence_length=sequence_length,
        num_episodes_per_iteration=num_episodes_per_iteration,
        num_epochs=num_epochs,
        per_replica_batch_size=per_replica_batch_size,
        num_replicas_in_sync=num_replicas_in_sync,
    )

    # 100 / 10 / 1 / 2 * 2 * 3 = 30
    self.assertEqual(init_iteration, 30)

  def test_compute_total_training_step(self):
    # The following parameters are mulipliers.
    sequence_length = 10
    num_iterations = 2
    num_episodes_per_iteration = 1
    num_epochs = 2

    # The following parameters are divisors.
    per_replica_batch_size = 2
    num_replicas_in_sync = 2

    total_training_step = train_ppo_lib.compute_total_training_step(
        sequence_length=sequence_length,
        num_iterations=num_iterations,
        num_episodes_per_iteration=num_episodes_per_iteration,
        num_epochs=num_epochs,
        per_replica_batch_size=per_replica_batch_size,
        num_replicas_in_sync=num_replicas_in_sync,
    )

    # 10 * 2 * 1 * 2 / 2 / 2 = 10
    self.assertEqual(total_training_step, 10)


if __name__ == '__main__':
  test_utils.main()
