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
"""Utility to create circuit learner."""

from typing import Callable, List, Optional, Text, Tuple

from absl import logging

import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.train import interval_trigger
from tf_agents.train import learner
from tf_agents.typing import types

# A function which processes a tuple of a nested tensor representing a TF-Agent
# Trajectory and Reverb SampleInfo.
_SequenceParamsType = Tuple[types.NestedTensor, types.ReverbSampleInfo]
_SequenceFnType = Callable[[_SequenceParamsType], _SequenceParamsType]


class CircuittrainingPPOLearner(object):
  """Manages all the learning details needed.

  These include:
    * Using distribution strategies correctly
    * Summaries
    * Checkpoints
    * Minimizing entering/exiting TF context:
        Especially in the case of TPUs scheduling a single TPU program to
        perform multiple train steps is critical for performance.
    * Generalizes the train call to be done correctly across CPU, GPU, or TPU
      executions managed by DistributionStrategies. This uses `strategy.run` and
      then makes sure to do a reduce operation over the `LossInfo` returned by
      the agent.
  """

  def __init__(
      self,
      root_dir: Text,
      train_step: tf.Variable,
      model_id: tf.Variable,
      agent: ppo_agent.PPOAgent,
      experience_dataset_fn: Callable[[], tf.data.Dataset],
      sequence_length: int,
      num_episodes_per_iteration: int,
      minibatch_size: int,
      shuffle_buffer_size: int,
      num_epochs: int = 1,
      triggers: Optional[List[interval_trigger.IntervalTrigger]] = None,
      checkpoint_interval: int = 100000,
      summary_interval: int = 1000,
      strategy: Optional[tf.distribute.Strategy] = None,
      per_sequence_fn: Optional[_SequenceFnType] = None,
      allow_variable_length_episodes: bool = False) -> None:
    """Initializes a CircuittrainingPPOLearner instance.

    Args:
      root_dir: Main directory path where checkpoints, saved_models, and
        summaries will be written to.
      train_step: a scalar tf.int64 `tf.Variable` which will keep track of the
        number of train steps. This is used for artifacts created like
        summaries, or outputs in the root_dir.
      model_id: a scalar tf.int64 `tf.Variable` which will keep track of the
        number of learner iterations / policy updates.
      agent: `ppo_agent.PPOAgent` instance to train with. Note that
        update_normalizers_in_train should be set to `False`, otherwise a
        ValueError will be raised. We do not update normalizers in the agent
        again because we already update it in the learner. When mini batching is
        enabled, compute_value_and_advantage_in_train should be set to False,
        and preprocessing should be done as part of the data pipeline as part of
        `replay_buffer.as_dataset`.
      experience_dataset_fn: a function that will create an instance of a
        tf.data.Dataset used to sample experience for training. Each element in
        the dataset is a (Trajectory, SampleInfo) pair.
      sequence_length: Fixed sequence length for elements in the dataset. Used
        for calculating how many iterations of minibatches to use for training.
      num_episodes_per_iteration: The number of episodes to sample for training.
        If fewer than this amount of episodes exists in the dataset, the learner
        will wait for more data to be added, or until the reverb timeout is
        reached.
      minibatch_size: The minibatch size. The dataset used for training is
        shaped `[minibatch_size, 1, ...]`. If None, full sequences will be fed
        into the agent. Please set this parameter to None for RNN networks which
        requires full sequences.
      shuffle_buffer_size: The buffer size for shuffling the trajectories before
        splitting them into mini batches. Only required when mini batch learning
        is enabled (minibatch_size is set). Otherwise it is ignored. Commonly
        set to a number 1-3x the episode length of your environment.
      num_epochs: The number of iterations to go through the same sequences.
      triggers: List of callables of the form `trigger(train_step)`. After every
        `run` call every trigger is called with the current `train_step` value
        as an np scalar.
      checkpoint_interval: Number of train steps in between checkpoints. Note
        these are placed into triggers and so a check to generate a checkpoint
        only occurs after every `run` call. Set to -1 to disable (this is not
        recommended, because it means that if the pipeline gets preempted, all
        previous progress is lost). This only takes care of the checkpointing
        the training process.  Policies must be explicitly exported through
        triggers.
      summary_interval: Number of train steps in between summaries. Note these
        are placed into triggers and so a check to generate a checkpoint only
        occurs after every `run` call.
      strategy: (Optional) `tf.distribute.Strategy` to use during training.
      per_sequence_fn: (Optional): sequence-wise preprecessing, pass in agent.
        preprocess for advantage calculation. This operation happens after
        take() and before rebatching.
      allow_variable_length_episodes: Whether to support variable length
        episodes for training.

    Raises:
      ValueError: agent._compute_value_and_advantage_in_train is set to `True`.
        preprocessing must be done as part of the data pipeline when mini
        batching is enabled.
    """

    strategy = strategy or tf.distribute.get_strategy()
    self._agent = agent
    self._minibatch_size = minibatch_size
    self._shuffle_buffer_size = shuffle_buffer_size
    self._num_epochs = num_epochs
    self._experience_dataset_fn = experience_dataset_fn
    self._num_episodes_per_iteration = num_episodes_per_iteration
    # Tracks the number of times learner.run() has been called.
    # This is used for filtering out data generated by older models to ensure
    # the on policyness of the algorithm.
    self._model_id = model_id
    self._sequence_length = sequence_length
    self._per_sequence_fn = per_sequence_fn

    self._generic_learner = learner.Learner(
        root_dir,
        train_step,
        agent,
        after_train_strategy_step_fn=None,
        triggers=triggers,
        checkpoint_interval=checkpoint_interval,
        summary_interval=summary_interval,
        use_kwargs_in_agent_train=False,
        strategy=strategy)

    self.num_replicas = strategy.num_replicas_in_sync
    self._allow_variable_length_episodes = allow_variable_length_episodes
    self._num_samples = self._num_episodes_per_iteration * self._sequence_length
    self._create_datasets(strategy)
    self._steps_per_iter = self._get_train_steps_per_iteration()
    logging.info('train steps per iteration: %d', self._steps_per_iter)

  def _create_datasets(self, strategy):
    """Create the training dataset and iterator."""

    def _filter_invalid_episodes(sample):
      sample_info = sample.info
      data_model_id = tf.cast(
          tf.reduce_min(sample_info.priority), dtype=tf.int64)

      if self._allow_variable_length_episodes:
        # Filter off policy samples.
        return tf.math.equal(self._model_id, data_model_id)
      else:
        # Filter infeasible placements with shorter episode lengths than
        # expected along with off policy samples.
        data = sample.data
        return tf.math.logical_and(
            tf.math.equal(tf.size(data.discount), self._sequence_length),
            tf.math.equal(self._model_id, data_model_id))

    def _make_dataset(_):
      # `experience_dataset_fn` returns a tf.Dataset. Each item is a (Trajectory
      # , SampleInfo) tuple, and the Trajectory represents one single episode
      # of a fixed sequence length. The Trajectory dimensions are [1, T, ...].
      train_dataset = self._experience_dataset_fn()
      train_dataset = train_dataset.filter(_filter_invalid_episodes)
      if not self._allow_variable_length_episodes:
        train_dataset = train_dataset.take(self._num_episodes_per_iteration)
      if self._per_sequence_fn:
        train_dataset = train_dataset.map(
            self._per_sequence_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

      # We take num_episodes_per_iteration, repeat for `num_epochs` times and
      # exhaust this data in the current learner run. The next time learner
      # runs, new batches of data will be sampled, cached and repeated.
      # This is enabled by the `Counter().flat_map()` trick below.

      # We unbatch the dataset shaped [B, T, ...] to a new dataset that
      # contains individual elements.
      # Note that we unbatch across the time dimension, which could result
      # in mini batches that contain subsets from more than one sequences.
      # PPO agent can handle mini batches across episode boundaries.
      train_dataset = train_dataset.unbatch()
      train_dataset = train_dataset.batch(1, drop_remainder=True).cache()

      if self._allow_variable_length_episodes:
        # Ideally we will train on num_episodes_per_iteration if all have the
        # max sequence_length, in case of shorter episodes we will train in an
        # equivalent number of steps num_episodes_per_iteration *
        # sequence_length.

        # Make sure we have enough samples to train on.
        train_dataset = train_dataset.take(self._num_samples).cache()

      train_dataset = train_dataset.shuffle(self._shuffle_buffer_size)
      train_dataset = train_dataset.repeat(self._num_epochs)
      train_dataset = train_dataset.batch(
          self._minibatch_size, drop_remainder=True)

      options = tf.data.Options()
      options.experimental_deterministic = False
      options.experimental_optimization.parallel_batch = True
      train_dataset = train_dataset.with_options(options)

      return train_dataset

    def make_dataset(_):
      return tf.data.experimental.Counter().flat_map(_make_dataset)

    with strategy.scope():

      if strategy.num_replicas_in_sync > 1:
        self._train_dataset = (
            strategy.distribute_datasets_from_function(make_dataset))
      else:
        self._train_dataset = make_dataset(0)
      self._train_iterator = iter(self._train_dataset)

  def _get_train_steps_per_iteration(self):
    """Number of train steps each time learner.run() is called."""

    # We exhaust all num_episodes_per_iteration taken from Reverb in this setup.
    # Here we assume that there's only 1 episode per batch, and each episode is
    # of the fixed sequence length.
    num_mini_batches = int(self._num_samples * self._num_epochs /
                           self._minibatch_size)
    train_steps = int(num_mini_batches / self.num_replicas)
    return train_steps

  def run(self):
    """Train `num_episodes_per_iteration` repeating for `num_epochs` of iterations.

    Returns:
      The total loss computed before running the final step.
    """
    loss_info = self._generic_learner.run(self._steps_per_iter,
                                          self._train_iterator)
    self._model_id.assign_add(1)

    return loss_info

  @property
  def train_step_numpy(self):
    """The current train_step.

    Returns:
      The current `train_step`. Note this will return a scalar numpy array which
      holds the `train_step` value when this was called.
    """
    return self._generic_learner.train_step_numpy
