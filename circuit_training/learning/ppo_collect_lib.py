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
"""Library for PPO collect job."""
import os
from typing import Any, Callable

from absl import logging
from circuit_training.learning import agent
from circuit_training.learning import static_feature_cache
from circuit_training.model import fully_connected_model_lib
from circuit_training.model import model
import reverb
import tensorflow as tf
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common


def collect(task: int,
            root_dir: str,
            replay_buffer_server_address: str,
            variable_container_server_address: str,
            create_env_fn: Callable[..., Any],
            max_sequence_length: int,
            write_summaries_task_threshold: int = 1,
            use_grl: bool = True,
            summary_subdir: str = ''):
  """Collects experience using a policy updated after every episode."""
  # Create the environment.
  train_step = train_utils.create_train_step()
  env = create_env_fn(train_step=train_step)
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

  policy = tf_agent.collect_policy
  tf_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_agent.collect_policy)

  # Create the variable container.
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.update(variables)

  # Create the replay buffer observer for collect jobs.
  observers = [
      reverb_utils.ReverbAddEpisodeObserver(
          reverb.Client(replay_buffer_server_address),
          table_name=['training_table'],
          max_sequence_length=max_sequence_length,
          priority=model_id)
  ]

  # Write metrics only if the task ID of the current job is below the limit.
  summary_dir = None
  metrics = []
  if task < write_summaries_task_threshold:
    summary_dir = os.path.join(root_dir, learner.TRAIN_DIR, summary_subdir,
                               str(task))
    metrics = actor.collect_metrics(1)

  # Create the collect actor.
  collect_actor = actor.Actor(
      env,
      tf_policy,
      train_step,
      episodes_per_run=1,
      summary_dir=summary_dir,
      metrics=metrics,
      observers=observers)

  # Run the experience collection loop.
  while True:
    collect_actor.run()
    variable_container.update(variables)
    logging.info('Collecting at step: %d', train_step.numpy())
    logging.info('Collecting at model_id: %d', model_id.numpy())
