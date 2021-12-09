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
r"""Tests for circuit training model.

# Internal circuit training docs link.
"""

import os

from absl import flags
from absl.testing import parameterized
from circuit_training.environment import environment
from circuit_training.model import model as grl_model
from circuit_training.utils import test_utils
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
import tf_agents.specs.tensor_spec as tensor_spec
from tf_agents.train.utils import strategy_utils
from tf_agents.trajectories import time_step as ts

flags.DEFINE_enum('strategy_type', 'cpu', [
    'tpu', 'gpu', 'cpu'
], ('Distribution Strategy type to use for training. `tpu` uses TPUStrategy for'
    ' running on TPUs (1x1), `gpu` uses GPUs with single host.'))
flags.DEFINE_integer(
    'global_batch_size', 64, 'Defines the global batch size. '
    'Note that for TPU the per TC batch size will be 32 for 1x1 TPU.')
flags.DEFINE_integer('dataset_repeat', 16,
                     'Defines the number of dataset repeat.')

FLAGS = flags.FLAGS

_TESTDATA_DIR = ('circuit_training/'
                 'environment/test_data')


class ActorModelTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ActorModelTest, self).setUp()
    block_name = 'sample_clustered'
    netlist_file = os.path.join(FLAGS.test_srcdir, _TESTDATA_DIR, block_name,
                                'netlist.pb.txt')
    init_placement = os.path.join(FLAGS.test_srcdir, _TESTDATA_DIR, block_name,
                                  'initial.plc')
    env = environment.create_circuit_environment(
        netlist_file=netlist_file, init_placement=init_placement)
    tf_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(env))
    self._input_tensors_spec = tf_env.observation_spec()
    self._output_tensors_spec = tf_env.action_spec()

    if FLAGS.strategy_type == 'tpu':
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      self._strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif FLAGS.strategy_type == 'gpu':
      self._strategy = strategy_utils.get_strategy(tpu=None, use_gpu=True)
    else:
      self._strategy = strategy_utils.get_strategy(tpu=None, use_gpu=False)

    with self._strategy.scope():
      shared_network = grl_model.GrlModel(
          input_tensors_spec=self._input_tensors_spec,
          output_tensors_spec=None,
          name='grl_model')
      self._value_model = grl_model.GrlValueModel(
          input_tensors_spec=self._input_tensors_spec,
          shared_network=shared_network,
          name='value_model')
      self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
      self._value_model.create_variables()

  def test_backwards_pass(self):
    observation_spec = self._input_tensors_spec
    time_step_spec = ts.time_step_spec(observation_spec)
    outer_dims = (FLAGS.global_batch_size,)
    time_step = tensor_spec.sample_spec_nest(
        time_step_spec, outer_dims=outer_dims)
    # TPU on forge has two cores (1x1).
    # The batch defined here represents the global batch size.
    # Will be evenly divided between the two cores.
    dataset = tf.data.Dataset.from_tensor_slices(time_step.observation).repeat(
        FLAGS.dataset_repeat).batch(FLAGS.global_batch_size)
    dist_dataset = self._strategy.experimental_distribute_dataset(dataset)
    with self._strategy.scope():

      def _step_fn(x):
        with tf.GradientTape() as tape:
          value, _ = self._value_model(x, training=True)
          loss = tf.math.reduce_sum(value)
        grads = tape.gradient(loss, self._value_model.trainable_variables)
        grads_and_vars = tuple(
            zip(grads, self._value_model.trainable_variables))
        self._optimizer.apply_gradients(grads_and_vars)

      @tf.function
      def _iter_fn(x):
        self._strategy.run(_step_fn, args=(x,))

    for x in dist_dataset:
      _iter_fn(x)


if __name__ == '__main__':
  test_utils.main()
