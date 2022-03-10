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
"""Testing the data collection."""

import os

from absl import flags
from absl import logging
from absl.testing import parameterized

from circuit_training.environment import environment
from circuit_training.learning import agent
from circuit_training.model import model

import numpy as np
import tensorflow.compat.v2 as tf

from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import array_spec
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils

FLAGS = flags.FLAGS

_TESTDATA_DIR = ('circuit_training/'
                 'environment/test_data/sample_clustered')


class _ValidateTimeStepObserver(object):
  """Observer that validates the time steps and collects episode lengths."""

  def __init__(self, test_case, time_step_spec, action_step_spec):
    self._test_case = test_case
    self._time_step_spec = time_step_spec
    self._action_step_spec = action_step_spec
    self._current_len = 0
    self._episode_lengths = []

  @property
  def episode_lengths(self):
    return self._episode_lengths

  def __call__(self, trajectory):
    # Check the time step spec.
    time_step = ts.TimeStep(
        trajectory.step_type,
        reward=trajectory.reward,
        discount=trajectory.discount,
        observation=trajectory.observation)
    logging.info('Time step: %s', time_step)
    logging.info('Time spec: %s', self._time_step_spec)
    self._test_case.assertTrue(
        array_spec.check_arrays_nest(time_step, self._time_step_spec))

    # Check the action step spec.
    action_step = policy_step.PolicyStep(
        action=trajectory.action, info=trajectory.policy_info)
    logging.info('Action step: %s', action_step.action)
    logging.info('Action spec: %s', self._action_step_spec)
    self._test_case.assertTrue(
        array_spec.check_arrays_nest(action_step.action,
                                     self._action_step_spec))

    # Update episode length statistics.
    logging.info('Index of trajector within the episode: %s', self._current_len)
    if trajectory.is_last():
      self._episode_lengths.append(self._current_len)
      self._current_len = 0
    else:
      self._current_len += 1


class CollectTest(parameterized.TestCase, test_utils.TestCase):

  @parameterized.named_parameters(
      ('_default', tf.distribute.get_strategy()),
      ('_one_device', tf.distribute.OneDeviceStrategy('/cpu:0')),
      ('_mirrored',
       tf.distribute.MirroredStrategy(devices=('/cpu:0', '/cpu:1'))))
  def test_collect_with_newly_initialized_ppo_collect_policy(self, strategy):
    # Create the environment.
    env = environment.create_circuit_environment(
        netlist_file=os.path.join(FLAGS.test_srcdir, _TESTDATA_DIR,
                                  'netlist.pb.txt'),
        init_placement=os.path.join(FLAGS.test_srcdir, _TESTDATA_DIR,
                                    'initial.plc'))
    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env))
    static_features = env.get_static_obs()
    grl_actor_net, grl_value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        static_features,
        strategy,
        use_model_tpu=False)

    # Create the agent whose collect policy is being tested.
    with strategy.scope():
      train_step = train_utils.create_train_step()
      tf_agent = agent.create_circuit_ppo_grl_agent(train_step,
                                                    action_tensor_spec,
                                                    time_step_tensor_spec,
                                                    grl_actor_net,
                                                    grl_value_net,
                                                    strategy)
      tf_agent.initialize()

    # Create, run driver and check the data in an observer performing asserts
    # the specs.
    validate_time_step = _ValidateTimeStepObserver(
        test_case=self,
        time_step_spec=env.time_step_spec(),
        action_step_spec=env.action_spec())
    driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(tf_agent.collect_policy),
        observers=[validate_time_step],
        max_episodes=10)
    driver.run(env.reset())

    # Make sure that environment steps were taken.
    self.assertLen(validate_time_step.episode_lengths, 10)
    episode_lens = np.array(validate_time_step.episode_lengths, dtype=np.int32)
    # Check if at least one of the rollouts took more than one step to ensure
    # that the time step validation has seen data.
    self.assertTrue(np.any(episode_lens > 1))
    logging.info('Observed episode lengths: %s', episode_lens)


if __name__ == '__main__':
  test_utils.main()
