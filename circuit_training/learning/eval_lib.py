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
from circuit_training.model import fully_connected_model_lib
from circuit_training.model import model
import tensorflow as tf
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import trajectory
from tf_agents.utils import common


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


def evaluate(root_dir: str,
             variable_container_server_address: str,
             create_env_fn: Callable[..., Any],
             use_grl: bool,
             extra_info_metrics: Optional[List[str]] = None,
             summary_subdir: str = ''):
  """Evaluates greedy policy."""

  # Create the variable container.
  train_step = train_utils.create_train_step()
  model_id = common.create_variable('model_id')

  # Create the environment.
  env = create_env_fn()
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env))
  static_features = env.wrapped_env().get_static_obs()
  cache = static_feature_cache.StaticFeatureCache()
  cache.add_static_feature(static_features)

  if use_grl:
    actor_net, value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        cache.get_all_static_features(),
        use_model_tpu=False)
    creat_agent_fn = agent.create_circuit_ppo_grl_agent
  else:
    actor_net = fully_connected_model_lib.create_actor_net(
        observation_tensor_spec, action_tensor_spec)
    value_net = fully_connected_model_lib.create_value_net(
        observation_tensor_spec)
    creat_agent_fn = agent.create_circuit_ppo_agent

  tf_agent = creat_agent_fn(
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
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.update(variables)

  # Create the evaluator actor.
  info_metrics = [
      InfoMetric(env, 'wirelength'),
      InfoMetric(env, 'congestion'),
      InfoMetric(env, 'density'),
  ]

  if extra_info_metrics:
    for info_metric in extra_info_metrics:
      info_metrics.append(InfoMetric(env, info_metric))

  eval_actor = actor.Actor(
      env,
      tf_policy,
      train_step,
      episodes_per_run=1,
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR, summary_subdir,
                               'eval'),
      metrics=[
          py_metrics.NumberOfEpisodes(),
          py_metrics.EnvironmentSteps(),
          py_metrics.AverageReturnMetric(
              name='eval_episode_return', buffer_size=1),
          py_metrics.AverageEpisodeLengthMetric(buffer_size=1),
      ] + info_metrics,
      name='performance')

  # Run the experience evaluation loop.
  while True:
    eval_actor.run()
    variable_container.update(variables)
    logging.info('Evaluating using greedy policy at step: %d',
                 train_step.numpy())
    # Write out summaries at the end of each evaluation iteration. This way,
    # we can look at the wirelength, density and congestion metrics more
    # frequently.
    eval_actor.write_metric_summaries()
    time.sleep(20)
