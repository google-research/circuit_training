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

import os
import time

from typing import Callable

from absl import logging

from circuit_training.environment import environment as oss_environment
from circuit_training.learning import agent
from circuit_training.learning import learner as learner_lib

import reverb
import tensorflow as tf

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.train import learner as actor_learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common


def train(
    root_dir: str,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    create_env_fn: Callable[[], oss_environment.CircuitEnv],
    sequence_length: int,
    actor_net: network.Network,
    value_net: network.Network,
    # Training params
    # This is the per replica batch size. The global batch size can be computed
    # by this number multiplied by the number of replicas (8 in the case of 2x2
    # TPUs).
    use_grl: bool = True,
    per_replica_batch_size: int = 32,
    num_epochs: int = 4,
    num_iterations: int = 10000,
    # This is the number of episodes we train on in each iteration.
    # num_episodes_per_iteration * epsisode_length * num_epochs =
    # global_step (number of gradient updates) * per_replica_batch_size *
    # num_replicas.
    num_episodes_per_iteration: int = 1024,
    allow_variable_length_episodes: bool = False) -> None:
  """Trains a PPO agent.

  Args:
    root_dir: Main directory path where checkpoints, saved_models, and summaries
      will be written to.
    strategy: `tf.distribute.Strategy` to use during training.
    replay_buffer_server_address: Address of the reverb replay server.
    variable_container_server_address: The address of the Reverb server for
      ReverbVariableContainer.
    create_env_fn: Function to create circuit training environment.
    sequence_length: Fixed sequence length for elements in the dataset. Used for
      calculating how many iterations of minibatches to use for training.
    actor_net: TF-Agents actor network.
    value_net: TF-Agents value network.
    use_grl: Whether to use GRL agent network or RL fully connected agent
      network.
    per_replica_batch_size: The minibatch size for learner. The dataset used for
      training is shaped `[minibatch_size, 1, ...]`. If None, full sequences
      will be fed into the agent. Please set this parameter to None for RNN
      networks which requires full sequences.
    num_epochs: The number of iterations to go through the same sequences. The
      num_episodes_per_iteration are repeated for num_epochs times in a
      particular learner run.
    num_iterations: The number of iterations to run the training.
    num_episodes_per_iteration: This is the number of episodes we train in each
      epoch.
    allow_variable_length_episodes: Whether to support variable length episodes
      for training.
  """
  # Get the specs from the environment.
  env = create_env_fn()
  _, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env))

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    model_id = common.create_variable('model_id')

    if use_grl:
      logging.info('Using GRL agent networks.')
      creat_agent_fn = agent.create_circuit_ppo_grl_agent
    else:
      logging.info('Using RL fully connected agent networks.')
      creat_agent_fn = agent.create_circuit_ppo_agent

    tf_agent = creat_agent_fn(
        train_step,
        action_tensor_spec,
        time_step_tensor_spec,
        actor_net,
        value_net,
        strategy,
    )
    tf_agent.initialize()

  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weights.
  saved_model_dir = os.path.join(root_dir, actor_learner.POLICY_SAVED_MODEL_DIR)
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir,
      tf_agent,
      train_step,
      start=-num_episodes_per_iteration,
      interval=num_episodes_per_iteration)

  # Create the variable container.
  variables = {
      reverb_variable_container.POLICY_KEY: tf_agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.push(variables)

  # Create the replay buffer.
  reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
      tf_agent.collect_data_spec,
      sequence_length=None,
      table_name='training_table',
      server_address=replay_buffer_server_address)

  # Initialize the dataset.
  def experience_dataset_fn():
    get_dtype = lambda x: x.dtype
    get_shape = lambda x: (None,) + x.shape
    shapes = tf.nest.map_structure(get_shape, tf_agent.collect_data_spec)
    dtypes = tf.nest.map_structure(get_dtype, tf_agent.collect_data_spec)

    dataset = reverb.TrajectoryDataset(
        server_address=replay_buffer_server_address,
        table='training_table',
        dtypes=dtypes,
        shapes=shapes,
        # Menger uses learner_iterations_per_call (256). Using 8 here instead
        # because we do not need that much data in the buffer (they have to be
        # filtered out for the next iteration anyways). The rule of thumb is
        # 2-3x batch_size.
        max_in_flight_samples_per_worker=8,
        num_workers_per_iterator=-1,
        max_samples_per_stream=-1,
        rate_limiter_timeout_ms=-1,
    )

    def broadcast_info(info_traj):
      # Assumes that the first element of traj is shaped
      # (sequence_length, ...); and we extract this length.
      info, traj = info_traj
      first_elem = tf.nest.flatten(traj)[0]
      length = first_elem.shape[0] or tf.shape(first_elem)[0]
      info = tf.nest.map_structure(lambda t: tf.repeat(t, [length]), info)
      return reverb.ReplaySample(info, traj)

    dataset = dataset.map(broadcast_info)
    return dataset

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=1000),
  ]

  def per_sequence_fn(sample):
    # At this point, each sample data contains a sequence of trajectories.
    data, info = sample.data, sample.info
    data = tf_agent.preprocess_sequence(data)
    return data, info

  learner = learner_lib.CircuittrainingPPOLearner(
      root_dir,
      train_step,
      model_id,
      tf_agent,
      experience_dataset_fn,
      sequence_length,
      num_episodes_per_iteration=num_episodes_per_iteration,
      minibatch_size=per_replica_batch_size,
      shuffle_buffer_size=(num_episodes_per_iteration * sequence_length),
      triggers=learning_triggers,
      summary_interval=1000,
      strategy=strategy,
      num_epochs=num_epochs,
      per_sequence_fn=per_sequence_fn,
      allow_variable_length_episodes=allow_variable_length_episodes)

  # Run the training loop.
  for i in range(num_iterations):
    step_val = train_step.numpy()
    logging.info('Training. Iteration: %d', i)
    start_time = time.time()
    learner.run()
    num_steps = train_step.numpy() - step_val
    run_time = time.time() - start_time
    logging.info('Steps per sec: %s', num_steps / run_time)
    logging.info('Pushing variables at model_id: %d', model_id.numpy())
    variable_container.push(variables)
    logging.info('clearing replay buffer')
    reverb_replay_train.clear()
