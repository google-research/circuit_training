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
from typing import Text

from absl import logging
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.policies import greedy_policy  # pylint: disable=unused-import
from tf_agents.policies import py_tf_eager_policy
from tf_agents.train import actor
from tf_agents.train import learner
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
      info_metric_key: a string key from the environment info to report,
        e.g. wirelength, density, congestion.
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


def evaluate(root_dir,
             variable_container_server_address,
             create_env_fn,
             extra_info_metrics=None):
  """Evaluates greedy policy."""

  # Create the path for the serialized greedy policy.
  policy_saved_model_path = os.path.join(root_dir,
                                         learner.POLICY_SAVED_MODEL_DIR,
                                         learner.GREEDY_POLICY_SAVED_MODEL_DIR)
  saved_model_pb_path = os.path.join(policy_saved_model_path, 'saved_model.pb')
  try:
    # Wait for the greedy policy to be outputed by learner (timeout after 2
    # days), then load it.
    train_utils.wait_for_file(
        saved_model_pb_path, sleep_time_secs=2, num_retries=86400)
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        policy_saved_model_path, load_specs_from_pbtxt=True)
  except TimeoutError as e:
    # If the greedy policy does not become available during the wait time of
    # the call `wait_for_file`, that probably means the learner is not running.
    logging.error('Could not get the file %s. Exiting.', saved_model_pb_path)
    raise e

  # Create the variable container.
  train_step = train_utils.create_train_step()
  model_id = common.create_variable('model_id')

  # Create the environment.
  env = create_env_fn()
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
      policy,
      train_step,
      episodes_per_run=1,
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR, 'eval'),
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
