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
"""Tests for circuit_training.model.model_lib."""

# TODO(b/219815138): Create a test verifying TPU model behaves the same as GPU.

from absl import flags
from absl import logging
from circuit_training.environment import observation_config
from circuit_training.model import model_lib
from circuit_training.utils import test_utils
import tensorflow as tf
from tf_agents.train.utils import strategy_utils

flags.DEFINE_enum('strategy_type', 'tpu', [
    'tpu', 'gpu', 'cpu'
], ('Distribution Strategy type to use for training. `tpu` uses TPUStrategy for'
    ' running on TPUs (1x1), `gpu` uses GPUs with single host.'))

FLAGS = flags.FLAGS


def make_strategy():
  if FLAGS.strategy_type == 'tpu':
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver('')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)
  elif FLAGS.strategy_type == 'gpu':
    return strategy_utils.get_strategy(tpu=None, use_gpu=True)
  else:
    return strategy_utils.get_strategy(tpu=None, use_gpu=False)


class ModelTest(test_utils.TestCase):

  def test_extract_feature(self):
    config = observation_config.ObservationConfig()
    static_features = config.observation_space.sample()
    strategy = make_strategy()
    with strategy.scope():
      if isinstance(strategy, tf.distribute.TPUStrategy):
        test_model = model_lib.CircuitTrainingTPUModel(
            static_features=static_features)
      else:
        test_model = model_lib.CircuitTrainingModel(
            static_features=static_features)

    @tf.function
    def forward():
      obs = config.observation_space.sample()
      obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), obs)
      return test_model(obs)

    per_replica_result = strategy.run(forward)
    logits, value = strategy.reduce('MEAN', per_replica_result, axis=None)

    logging.info('logits: %s', logits)
    logging.info('value: %s', value)
    self.assertAllEqual(logits['location'].shape, (1, config.max_grid_size**2))
    self.assertAllEqual(value.shape, (1, 1))

  def test_backwards_pass(self):
    config = observation_config.ObservationConfig()
    static_features = config.observation_space.sample()
    strategy = make_strategy()
    with strategy.scope():
      if isinstance(strategy, tf.distribute.TPUStrategy):
        test_model = model_lib.CircuitTrainingTPUModel(
            static_features=static_features)
      else:
        test_model = model_lib.CircuitTrainingModel(
            static_features=static_features)
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    obs = config.observation_space.sample()
    obs = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), obs)

    @tf.function
    def loss_fn(x, training=False):
      logits, value = test_model(x, training=training)
      loss = (
          tf.math.reduce_sum(logits['location']) + tf.math.reduce_sum(value))
      return loss

    def train_step(obs):
      with tf.GradientTape() as tape:
        loss = loss_fn(obs, training=True)
        grads = tape.gradient(loss, test_model.trainable_variables)
      optimizer.apply_gradients(zip(grads, test_model.trainable_variables))
      return loss

    @tf.function
    def loss_fn_run(obs):
      loss = strategy.run(loss_fn, args=(obs,))
      return strategy.reduce('MEAN', loss, axis=None)

    @tf.function
    def train_step_run(obs):
      strategy.run(train_step, args=(obs,))

    # Gather variables and loss before training
    initial_loss = loss_fn_run(obs).numpy()
    initial_weights = [v.numpy() for v in test_model.trainable_variables]

    # Run one train step
    train_step_run(obs)
    # Re-compute the loss
    current_loss = loss_fn_run(obs).numpy()
    current_weights = [v.numpy() for v in test_model.trainable_variables]
    # Verify loss and weights have changed.
    self.assertNotAllClose(initial_weights, current_weights)
    self.assertNotAlmostEqual(initial_loss, current_loss)


if __name__ == '__main__':
  test_utils.main()
