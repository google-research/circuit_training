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
"""Tests for circuit_training.environment."""

import os

from absl import flags
from circuit_training.environment import environment
from circuit_training.utils import test_utils
import gin
import numpy as np
import tensorflow as tf
from tf_agents import specs
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts

# Internal gfile dependencies

FLAGS = flags.FLAGS


def random_action(mask):
  valid_actions, = np.nonzero(mask.flatten())
  if len(valid_actions):  # pylint: disable=g-explicit-length-test
    return np.random.choice(valid_actions)

  # If there is no valid choice, then `[0]` is returned which results in an
  # infeasable action ending the episode.
  return 0


class _RandomValidCircuitPolicy(random_py_policy.RandomPyPolicy):
  """Policy wrapper for the function `random_action(mask)` above."""

  def _action(self, time_step, policy_state):
    valid_random_action = random_action(time_step.observation['mask'])
    return policy_step.PolicyStep(
        action=valid_random_action, state=policy_state)


class _ValidateTimeStepObserver(object):
  """Observer that validates the time steps and collects episode lengths."""

  def __init__(self, test_case, time_step_spec):
    self._test_case = test_case
    self._time_step_spec = time_step_spec
    self._current_len = 0
    self._episode_lengths = []

  @property
  def episode_lengths(self):
    return self._episode_lengths

  def __call__(self, trajectory):
    time_step = ts.TimeStep(
        trajectory.step_type,
        reward=trajectory.reward,
        discount=trajectory.discount,
        observation=trajectory.observation)
    if trajectory.is_last():
      self._episode_lengths.append(self._current_len)
      self._current_len = 0
    else:
      self._current_len += 1
    self._test_case.assertTrue(
        array_spec.check_arrays_nest(time_step, self._time_step_spec))


def infeasible_action(mask):
  return np.random.choice(np.nonzero(1 - mask.flatten())[0])


class EnvironmentTest(test_utils.TestCase):
  """Tests for the Environment.

  # Internal circuit training docs link.
  """

  def test_create_and_obs_space(self):
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/sample_clustered')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    env = environment.CircuitEnv(
        netlist_file=netlist_file, init_placement=init_placement)

    obs = env.reset()
    self.assertTrue(env.observation_space.contains(obs))
    done = False
    while not done:
      action = random_action(obs['mask'])
      obs, reward, done, _ = env.step(action)
      self.assertTrue(env.observation_space.contains(obs))
      self.assertIsInstance(reward, float)
      self.assertIsInstance(done, bool)

  def test_save_file_train_step(self):
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/sample_clustered')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    output_dir = self.create_tempdir()
    output_plc_file = os.path.join(output_dir, 'ppo_opt_placement.plc')
    output_cd_file = os.path.join(output_dir, 'ppo_cd_placement.plc')

    train_step = train_utils.create_train_step()
    train_step.assign(1234)

    env = environment.CircuitEnv(
        netlist_file=netlist_file,
        init_placement=init_placement,
        is_eval=True,
        save_best_cost=True,
        output_plc_file=output_plc_file,
        cd_finetune=True,
        train_step=train_step)

    obs = env.reset()
    done = False
    while not done:
      action = random_action(obs['mask'])
      obs, _, done, _ = env.step(action)

    self.assertTrue(os.path.exists(output_plc_file))
    with open(output_plc_file) as f:
      self.assertIn('Train step : 1234', f.read())
    self.assertTrue(os.path.exists(output_cd_file))
    with open(output_cd_file) as f:
      self.assertIn('Train step : 1234', f.read())

  def test_action_space(self):
    bindings = """
      ObservationConfig.max_grid_size = 128
    """
    gin.parse_config(bindings)
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/sample_clustered')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    env = environment.CircuitEnv(
        netlist_file=netlist_file, init_placement=init_placement)
    self.assertEqual(env.action_space.shape, ())
    self.assertTrue(env.action_space.contains(0))
    self.assertTrue(env.action_space.contains(128**2 - 1))
    self.assertFalse(env.action_space.contains(128**2))

    mask = env.reset()['mask']

    # Outside of the real canvas:
    self.assertFalse(mask[0])
    self.assertFalse(mask[-1])

    # Inside of the real canvas:
    up_pad = (128 - 2) // 2
    right_pad = (128 - 2) // 2
    self.assertTrue(mask[(up_pad + 0) * 128 + (right_pad + 0)])  # (0, 0)
    self.assertTrue(mask[(up_pad + 1) * 128 + (right_pad + 1)])  # (1, 1)

  def test_infisible(self):
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/sample_clustered')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    env = environment.CircuitEnv(
        netlist_file=netlist_file,
        init_placement=init_placement,
    )
    obs = env.reset()
    action = random_action(obs['mask'])
    obs, _, _, _ = env.step(action)
    action = infeasible_action(obs['mask'])
    with self.assertRaises(environment.InfeasibleActionError):
      env.step(action)

  def test_wrap_tfpy_environment(self):
    bindings = """
      ObservationConfig.max_grid_size = 128
    """
    gin.parse_config(bindings)
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/sample_clustered')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    env = environment.create_circuit_environment(
        netlist_file=netlist_file,
        init_placement=init_placement,
    )
    tf_env = tf_py_environment.TFPyEnvironment(env)
    spec = tf_env.action_spec()
    self.assertEqual(type(spec), specs.BoundedTensorSpec)
    self.assertEqual(spec.dtype, tf.int64)
    self.assertEqual(spec.shape, ())
    self.assertEqual(spec.minimum, 0)
    self.assertEqual(spec.maximum, 128 * 128 - 1)
    self.assertEqual(spec.name, 'action')

  def test_validate_circuite_env(self):
    test_netlist_dir = ('circuit_training/'
                        'environment/test_data/sample_clustered')
    netlist_file = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, test_netlist_dir,
                                  'initial.plc')
    env = environment.create_circuit_environment(
        netlist_file=netlist_file,
        init_placement=init_placement,
    )

    # Create a Python policy that provides *valid* random actions.
    time_step_spec = env.time_step_spec()
    valid_random_policy = _RandomValidCircuitPolicy(
        time_step_spec=time_step_spec, action_spec=env.action_spec())

    # Create an observer that asserts that the time steps are valid given the
    # time step spec of the environment.
    validate_time_step = _ValidateTimeStepObserver(
        test_case=self, time_step_spec=time_step_spec)

    # Create and run a driver using to validate the time steps observerd.
    driver = py_driver.PyDriver(
        env,
        valid_random_policy,
        observers=[validate_time_step],
        max_episodes=10)
    driver.run(env.reset())

    # Make sure that environment steps were taken.
    self.assertLen(validate_time_step.episode_lengths, 10)
    episode_lens = np.array(validate_time_step.episode_lengths, dtype=np.int32)
    # Check if at least one of the rollouts took more than one step to ensure
    # that the time step validation has seen data.
    self.assertTrue(np.any(episode_lens > 1))


if __name__ == '__main__':
  test_utils.main()
