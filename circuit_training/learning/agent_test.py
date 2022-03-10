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
"""Tests for agent and network creation."""

import os

from absl import flags
from absl import logging
from circuit_training.environment import environment
from circuit_training.learning import agent
from circuit_training.model import model
from circuit_training.utils import test_utils

import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

FLAGS = flags.FLAGS

_TESTDATA_DIR = ('circuit_training/'
                 'environment/test_data/sample_clustered')


def create_test_circuit_env():
  env = environment.create_circuit_environment(
      netlist_file=os.path.join(
          FLAGS.test_srcdir, _TESTDATA_DIR, 'netlist.pb.txt'),
      init_placement=os.path.join(
          FLAGS.test_srcdir, _TESTDATA_DIR, 'initial.plc'))
  return env


class AgentTest(test_utils.TestCase):

  def test_value_network_grl(self):
    """GRL value network outputs the expected shape."""
    env = create_test_circuit_env()
    observation_tensor_spec, action_tensor_spec, _ = (
        spec_utils.get_tensor_specs(env))
    logging.info('action_tensor_spec: %s', action_tensor_spec)
    time_step_tensor_spec = ts.time_step_spec(observation_tensor_spec)

    train_step = train_utils.create_train_step()
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
    static_features = env.get_static_obs()

    observation_tensor_spec, action_tensor_spec, _ = (
        spec_utils.get_tensor_specs(env))
    grl_actor_net, grl_value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        static_features,
        strategy,
        use_model_tpu=False)

    grl_agent = agent.create_circuit_ppo_grl_agent(
        train_step,
        action_tensor_spec,
        time_step_tensor_spec,
        grl_actor_net,
        grl_value_net,
        strategy,
    )

    batch_size = 4
    # Check that value prediction outputs the correct shape (B, ).
    sample_time_steps = tensor_spec.sample_spec_nest(
        time_step_tensor_spec, outer_dims=(batch_size,))
    value_outputs, _ = grl_agent.collect_policy.apply_value_network(
        sample_time_steps.observation,
        sample_time_steps.step_type,
        value_state=(),
        training=False)
    self.assertEqual(value_outputs.shape, (batch_size,))

  def test_train_grl(self):
    """GRL training does not fail on arbitrary data."""
    env = create_test_circuit_env()
    observation_tensor_spec, action_tensor_spec, _ = (
        spec_utils.get_tensor_specs(env))
    logging.info('action_tensor_spec: %s', action_tensor_spec)
    time_step_tensor_spec = ts.time_step_spec(observation_tensor_spec)

    train_step = train_utils.create_train_step()
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
    static_features = env.get_static_obs()

    grl_actor_net, grl_value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        static_features,
        strategy,
        use_model_tpu=False)
    grl_agent = agent.create_circuit_ppo_grl_agent(
        train_step,
        action_tensor_spec,
        time_step_tensor_spec,
        grl_actor_net,
        grl_value_net,
        strategy)

    batch_size = 4
    sample_time_steps = tensor_spec.sample_spec_nest(
        time_step_tensor_spec, outer_dims=(batch_size, 1))
    sample_actions = tensor_spec.sample_spec_nest(
        action_tensor_spec, outer_dims=(batch_size, 1))
    sample_policy_info = {
        'dist_params': {
            'logits':
                tf.ones_like(
                    sample_time_steps.observation['mask'],
                    dtype=tf.dtypes.float32)
        },
        'value_prediction': tf.constant([[0.2]] * batch_size),
        'return': tf.constant([[0.2]] * batch_size),
        'advantage': tf.constant([[0.2]] * batch_size),
    }
    sample_experience = trajectory.Trajectory(
        sample_time_steps.step_type, sample_time_steps.observation,
        sample_actions, sample_policy_info, sample_time_steps.step_type,
        sample_time_steps.reward, sample_time_steps.discount)
    # Check that training compeltes one iteration.
    grl_agent.train(sample_experience)

if __name__ == '__main__':
  test_utils.main()
