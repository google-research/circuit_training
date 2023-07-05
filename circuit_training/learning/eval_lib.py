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
"""Eval job using a variable container to fetch the weights of the policy."""

import collections
import os
import statistics
import time
from typing import Any, Callable, List, Optional, Text

from absl import logging
from circuit_training.learning import agent
from circuit_training.learning import static_feature_cache
import numpy as np
import tensorflow as tf
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


class PlacementImage(py_metric.PyStepMetric):
  """Observer for recording the placement image."""

  def __init__(self, num_rows: int, num_cols: int):
    """Placement Image.

    Args:
      num_rows: number of rows of the policy image.
      num_cols: number of cols of the policy image.
    """
    super(PlacementImage, self).__init__('placement_image')

    self._num_rows = num_rows
    self._num_cols = num_cols
    self._locations = []

  def call(self, traj: trajectory.Trajectory) -> None:
    if traj.step_type == ts.StepType.FIRST:
      self._locations = []

    self._locations.append(traj.action)

  def result(self) -> np.ndarray:
    macro_locations = np.zeros((self._num_rows * self._num_cols,))
    macro_locations[self._locations] = (
        np.arange(1, len(self._locations) + 1, dtype=np.float32)
    ) / len(self._locations)
    return np.reshape(macro_locations, (1, self._num_rows, self._num_cols, 1))

  def reset(self) -> None:
    self._locations = []


class FirstPolicyImage(py_metric.PyStepMetric):
  """Observer for recording the first policy image."""

  def __init__(self, num_rows: int, num_cols: int, actor_net: network.Network):
    """First Policy Image.

    Args:
      num_rows: number of rows of the policy image.
      num_cols: number of cols of the policy image.
      actor_net: Actor net.
    """
    super(FirstPolicyImage, self).__init__('first_policy_image')

    self._num_rows = num_rows
    self._num_cols = num_cols
    self._actor_net = actor_net
    self._first_policy_image = np.zeros((self._num_rows, self._num_cols))

  def call(self, traj: trajectory.Trajectory) -> None:
    def normalize_image(image: np.ndarray) -> np.ndarray:
      max_val = np.amax(image)
      min_val = np.amin(image)
      return (image - min_val) / (max_val - min_val)

    if traj.step_type == ts.StepType.FIRST:
      obs = tf.nest.map_structure(
          lambda t: tf.expand_dims(t, 0), traj.observation
      )
      dist, _ = self._actor_net(obs)
      self._first_policy_image = normalize_image(
          tf.squeeze(dist.probs_parameter())
      )

  def result(self) -> np.ndarray:
    return np.reshape(
        self._first_policy_image, (1, self._num_rows, self._num_cols, 1)
    )

  def reset(self) -> None:
    self._first_policy_image = np.zeros((self._num_rows, self._num_cols))


class InfoMetric(py_metric.PyStepMetric):
  """Observer for graphing the environment info metrics."""

  def __init__(
      self,
      env,
      info_metric_key: Text,
      buffer_size: int = 1,
      name: Text = 'InfoMetric',
  ):
    """Observer reporting TensorBoard metrics at the end of each episode.

    Args:
      env: environment.
      info_metric_key: a string key from the environment info to report, e.g.
        wirelength, density, congestion.
      buffer_size: size of the buffer for calculating the aggregated metrics.
      name: name of the observer object.
    """
    super(InfoMetric, self).__init__(name + '_' + info_metric_key)

    self._env = env
    self._info_metric_key = info_metric_key
    self._buffer = collections.deque(maxlen=buffer_size)

  def call(self, traj: trajectory.Trajectory):
    """Report the requested metrics at the end of each episode."""

    # We collect the metrics from the info from the environment instead.
    # The traj argument is kept to be compatible with the actor/learner API
    # for metrics.
    del traj

    if self._env.done:
      metric_value = self._env.get_info()[self._info_metric_key]
      self._buffer.append(metric_value)

  def result(self):
    return statistics.mean(self._buffer)

  def reset(self):
    self._buffer.clear()


def evaluate(
    root_dir: str,
    variable_container_server_address: str,
    create_env_fn: Callable[..., Any],
    create_models_fn: Callable[..., Any],
    rl_architecture: str = 'generalization',
    info_metric_names: Optional[List[str]] = None,
    summary_subdir: str = '',
):
  """Evaluates greedy policy."""

  # Create the variable container.
  train_step = train_utils.create_train_step()
  model_id = common.create_variable('model_id')

  # Create the environment.
  env = create_env_fn(train_step=train_step)
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env)
  )
  static_features = env.wrapped_env().get_static_obs()
  cache = static_feature_cache.StaticFeatureCache()
  cache.add_static_feature(static_features)

  actor_net, value_net = create_models_fn(
      rl_architecture,
      observation_tensor_spec,
      action_tensor_spec,
      cache.get_all_static_features(),
  )

  if rl_architecture == 'static_graph_embedding':
    image_metrics = [
        PlacementImage(env.grid_rows, env.grid_cols),
        FirstPolicyImage(env.grid_rows, env.grid_cols, actor_net),
    ]
  else:
    image_metrics = [
        PlacementImage(
            env.observation_config.max_grid_size,
            env.observation_config.max_grid_size,
        ),
        FirstPolicyImage(
            env.observation_config.max_grid_size,
            env.observation_config.max_grid_size,
            actor_net,
        ),
    ]

  tf_agent = agent.create_circuit_ppo_agent(
      train_step,
      action_tensor_spec,
      time_step_tensor_spec,
      actor_net,
      value_net,
      tf.distribute.get_strategy(),
  )

  policy = greedy_policy.GreedyPolicy(tf_agent.policy)
  tf_policy = py_tf_eager_policy.PyTFEagerPolicy(policy)

  variables = {
      reverb_variable_container.POLICY_KEY: policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE],
  )
  variable_container.update(variables)

  # Create the evaluator actor.
  eval_actor = actor.Actor(
      env,
      tf_policy,
      train_step,
      episodes_per_run=1,
      summary_dir=os.path.join(
          root_dir, learner.TRAIN_DIR, summary_subdir, 'eval'
      ),
      metrics=[
          py_metrics.NumberOfEpisodes(),
          py_metrics.EnvironmentSteps(),
          py_metrics.AverageReturnMetric(
              name='eval_episode_return', buffer_size=1
          ),
          py_metrics.AverageEpisodeLengthMetric(buffer_size=1),
      ]
      + [InfoMetric(env, info_metric) for info_metric in info_metric_names],
      image_metrics=image_metrics,
      name='performance',
  )

  # Run the experience evaluation loop.
  while True:
    logging.info(
        'Evaluating using greedy policy at step: %d', train_step.numpy()
    )
    eval_actor.run()
    logging.info('Updating the variables.')
    variable_container.update(variables)
    # Write out summaries at the end of each evaluation iteration. This way,
    # we can look at the wirelength, density and congestion metrics more
    # frequently.
    logging.info('Updating the summaries.')
    eval_actor.write_metric_summaries()
    time.sleep(20)
