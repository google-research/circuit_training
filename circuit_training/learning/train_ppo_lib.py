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

from absl import logging
from circuit_training.learning import agent
from circuit_training.learning import learner as learner_lib
import gin
import reverb
import tensorflow as tf
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.train import learner as actor_learner
from tf_agents.train import triggers
from tf_agents.train.utils import train_utils
from tf_agents.typing import types
from tf_agents.utils import common


@gin.configurable(allowlist=['shuffle_buffer_episode_len'])
def get_shuffle_buffer_size(
    sequence_length: int,
    shuffle_buffer_episode_len: int = 3,
) -> int:
  """Returns shuffle buffer size.

  Args:
    sequence_length: The sequence length.
    shuffle_buffer_episode_len: The size of buffer for shuffle operation in
      dataset. The buffer size should be between 1-3 episode len.

  Returns:
    The shuffle buffer size.
  """
  return sequence_length * shuffle_buffer_episode_len


def compute_init_iteration(
    init_train_step: int,
    sequence_length: int,
    num_episodes_per_iteration: int,
    num_epochs: int,
    per_replica_batch_size: int,
    num_replicas_in_sync: int,
) -> int:
  """Computes the initial iterations number.

  In case of restarting, the init_train_step might not be zero. We need to
  compute the initial iteration number to offset the total number of iterations.

  Args:
    init_train_step: Initial train step.
    sequence_length: Fixed sequence length for elements in the dataset. Used for
      calculating how many iterations of minibatches to use for training.
    num_episodes_per_iteration: This is the number of episodes we train in each
      epoch.
    num_epochs: The number of iterations to go through the same sequences. The
      num_episodes_per_iteration are repeated for num_epochs times in a
      particular learner run.
    per_replica_batch_size: The minibatch size for learner. The dataset used for
      training is shaped `[minibatch_size, 1, ...]`. If None, full sequences
      will be fed into the agent. Please set this parameter to None for RNN
      networks which requires full sequences.
    num_replicas_in_sync: The number of replicas training in sync.

  Returns:
    The initial iteration number.
  """
  return int(
      init_train_step
      * per_replica_batch_size
      * num_replicas_in_sync
      / sequence_length
      / num_episodes_per_iteration
      / num_epochs
  )


def compute_total_training_step(
    sequence_length,
    num_iterations,
    num_episodes_per_iteration,
    num_epochs,
    per_replica_batch_size,
    num_replicas_in_sync,
) -> int:
  """Computes the total training step.

  Args:
    sequence_length: Fixed sequence length for elements in the dataset. Used for
      calculating how many iterations of minibatches to use for training.
    num_iterations: The number of iterations to run the training.
    num_episodes_per_iteration: This is the number of episodes we train in each
      epoch.
    num_epochs: The number of iterations to go through the same sequences. The
      num_episodes_per_iteration are repeated for num_epochs times in a
      particular learner run.
    per_replica_batch_size: The minibatch size for learner. The dataset used for
      training is shaped `[minibatch_size, 1, ...]`. If None, full sequences
      will be fed into the agent. Please set this parameter to None for RNN
      networks which requires full sequences.
    num_replicas_in_sync: The number of replicas training in sync.

  Returns:
    The total training step.
  """
  return int(
      sequence_length
      * num_iterations
      * num_episodes_per_iteration
      * num_epochs
      / per_replica_batch_size
      / num_replicas_in_sync
  )


@gin.configurable(
    allowlist=[
        'per_replica_batch_size',
        'num_epochs',
        'num_iterations',
        'num_episodes_per_iteration',
        'init_learning_rate',
        'policy_save_interval',
    ]
)
def train(
    root_dir: str,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: types.NestedTensorSpec,
    sequence_length: int,
    actor_net: network.Network,
    value_net: network.Network,
    # Training params
    init_train_step: int = 0,
    # This is the per replica batch size. The global batch size can be computed
    # by this number multiplied by the number of replicas (8 in the case of 2x2
    # TPUs).
    per_replica_batch_size: int = 128,
    num_epochs: int = 4,
    # Set to a very large number so the learning rate remains the same, and
    # also the deadline stops the training rather than this param.
    num_iterations: int = 1_000_000_000,
    # This is the number of episodes we train on in each iteration.
    # num_episodes_per_iteration * epsisode_length * num_epochs =
    # global_step (number of gradient updates) * per_replica_batch_size *
    # num_replicas.
    num_episodes_per_iteration: int = 256,
    init_learning_rate: float = 0.004,
    policy_save_interval: int = 1000,
    num_netlists: int = 1,
    debug_summaries: bool = False,
) -> None:
  """Trains a PPO agent.

  Args:
    root_dir: Main directory path where checkpoints, saved_models, and summaries
      will be written to.
    strategy: `tf.distribute.Strategy` to use during training.
    replay_buffer_server_address: Address of the reverb replay server.
    variable_container_server_address: The address of the Reverb server for
      ReverbVariableContainer.
    action_tensor_spec: Action tensor_spec.
    time_step_tensor_spec: Time step tensor_spec.
    sequence_length: Fixed sequence length for elements in the dataset. Used for
      calculating how many iterations of minibatches to use for training.
    actor_net: TF-Agents actor network.
    value_net: TF-Agents value network.
    init_train_step: Initial train step.
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
    init_learning_rate: Initial learning rate.
    policy_save_interval: How often policies are saved.
    num_netlists: Number of netlits to train used for normalizing advantage. If
      larger than 1, the advantage will be normalize first across the netlists
      then on the entire batch.
    debug_summaries: If enable summray extra information.
  """

  init_iteration = compute_init_iteration(
      init_train_step,
      sequence_length,
      num_episodes_per_iteration,
      num_epochs,
      per_replica_batch_size,
      strategy.num_replicas_in_sync,
  )
  logging.info('Initialize iteration at: init_iteration %s.', init_iteration)

  total_training_step = compute_total_training_step(
      sequence_length,
      num_iterations,
      num_episodes_per_iteration,
      num_epochs,
      per_replica_batch_size,
      strategy.num_replicas_in_sync,
  )

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    train_step.assign(init_train_step)
    logging.info('Initialize train_step at %s', init_train_step)
    model_id = common.create_variable('model_id')
    # The model_id should equal to the iteration number.
    model_id.assign(init_iteration)

    lr = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=init_learning_rate,
        decay_steps=total_training_step,
        alpha=0.1,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-5)
    # Assigns the train step to optimizer iterations to ensure that the step is
    # correct when resuming training.
    optimizer.iterations = train_step

    tf_agent = agent.create_circuit_ppo_agent(
        train_step=train_step,
        action_tensor_spec=action_tensor_spec,
        time_step_tensor_spec=time_step_tensor_spec,
        actor_net=actor_net,
        value_net=value_net,
        strategy=strategy,
        optimizer=optimizer,
    )
    tf_agent.initialize()

  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weights.
  saved_model_dir = os.path.join(root_dir, actor_learner.POLICY_SAVED_MODEL_DIR)
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir,
      tf_agent,
      train_step,
      start=-policy_save_interval,
      interval=policy_save_interval,
  )

  # Create the variable container.
  variables = {
      reverb_variable_container.POLICY_KEY: tf_agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE],
  )
  variable_container.push(variables)

  # Create the replay buffer.
  reverb_replay_trains = []
  for index in range(num_netlists):
    reverb_replay_trains += [
        reverb_replay_buffer.ReverbReplayBuffer(
            tf_agent.collect_data_spec,
            sequence_length=None,
            table_name=f'training_table_{index}',
            server_address=replay_buffer_server_address,
        )
    ]

  # Initialize the dataset.
  def experiences_dataset_fn():
    get_dtype = lambda x: x.dtype
    get_shape = lambda x: (None,) + x.shape
    shapes = tf.nest.map_structure(get_shape, tf_agent.collect_data_spec)
    dtypes = tf.nest.map_structure(get_dtype, tf_agent.collect_data_spec)

    def broadcast_info(info_traj):
      # Assumes that the first element of traj is shaped
      # (sequence_length, ...); and we extract this length.
      info, traj = info_traj
      first_elem = tf.nest.flatten(traj)[0]
      length = first_elem.shape[0] or tf.shape(first_elem)[0]
      info = tf.nest.map_structure(lambda t: tf.repeat(t, [length]), info)
      return reverb.ReplaySample(info, traj)

    datasets = []
    for index in range(num_netlists):
      dataset = reverb.TrajectoryDataset(
          server_address=replay_buffer_server_address,
          table=f'training_table_{index}',
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
      logging.info('Created dataset for training_table_%s', index)

      datasets += [dataset.map(broadcast_info)]

    return datasets

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=200),
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
      experiences_dataset_fn,
      sequence_length,
      num_episodes_per_iteration=num_episodes_per_iteration,
      minibatch_size=per_replica_batch_size,
      shuffle_buffer_size=get_shuffle_buffer_size(sequence_length),
      triggers=learning_triggers,
      strategy=strategy,
      num_epochs=num_epochs,
      per_sequence_fn=per_sequence_fn,
  )

  # Run the training loop.
  for i in range(init_iteration, num_iterations):
    step_val = train_step.numpy()
    logging.info('Training. Iteration: %d', i)
    start_time = time.time()
    if debug_summaries:
      # `wait_for_data` is not necessary and is added only to measure the data
      # latency. It takes one batch of data from dataset and print it. So, it
      # waits until the data is ready to consume.
      learner.wait_for_data()
      data_wait_time = time.time() - start_time
      logging.info('Data wait time sec: %s', data_wait_time)
    learner.run()
    run_time = time.time() - start_time
    num_steps = train_step.numpy() - step_val
    logging.info('Steps per sec: %s', num_steps / run_time)
    logging.info('Pushing variables at model_id: %d', model_id.numpy())
    variable_container.push(variables)
    logging.info('clearing replay buffers')
    for reverb_replay_train in reverb_replay_trains:
      reverb_replay_train.clear()
    with (
        learner.train_summary_writer.as_default(),
        common.soft_device_placement(),
        tf.summary.record_if(lambda: True),
    ):
      with tf.name_scope('RunTime/'):
        tf.summary.scalar(
            name='step_per_sec', data=num_steps / run_time, step=train_step
        )
        if debug_summaries:
          tf.summary.scalar(
              name='data_wait_time_sec', data=data_wait_time, step=train_step
          )
