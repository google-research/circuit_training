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
"""Helper functions for creating the fully connected models.

This model architecture creates a simple agent which can't generealize
over multiple netlists, but it has a low inference and train cost which makes it
more suitable that the GCN-based model for reward function development.
"""

import functools
from typing import Optional

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.typing import types


def create_actor_net(
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    seed: Optional[types.Seed] = None) -> sequential.Sequential:
  """Define the actor network."""
  seed_stream = tfp.util.SeedStream(seed=seed, salt='weight_init_seed')
  init = tf.keras.initializers.GlorotUniform(seed=seed_stream())

  dense = functools.partial(
      tf.keras.layers.Dense, activation='relu', kernel_initializer=init)
  fc_layer_units = [64, 64, 64, 64]

  def no_op_layer():
    return tf.keras.layers.Lambda(lambda x: x)

  def projection_layer():
    return tf.keras.layers.Dense(
        np.unique(action_tensor_spec.maximum - action_tensor_spec.minimum + 1),
        activation=None,
        kernel_initializer=init,
        name='projection_layer')

  def create_dist(logits_and_mask):
    # Apply mask onto the logits such that infeasible actions will not be taken.
    logits, mask = logits_and_mask.values()

    if mask.shape.rank < logits.shape.rank:
      mask = tf.expand_dims(mask, -2)

    # Overwrite the logits for invalid (!= 1) actions to a very large negative
    # number. We do not use -inf because it produces NaNs in many tfp
    # functions.
    # Currently keep aligned with Menger. Eventually move to logits.dtype.min.
    almost_neg_inf = tf.ones_like(logits) * (-2.**32 + 1)
    logits = tf.where(tf.equal(mask, 1), logits, almost_neg_inf)

    return tfp.distributions.Categorical(
        logits=logits, dtype=action_tensor_spec.dtype)

  return sequential.Sequential(
      [
          nest_map.NestMap({
              'graph_embedding':
                  tf.keras.Sequential(
                      [tf.keras.layers.Flatten()] +
                      [dense(num_units)
                       for num_units in fc_layer_units] + [projection_layer()]),
              'mask':
                  no_op_layer(),
          })
      ] +
      # Create the output distribution from the mean and standard deviation.
      [tf.keras.layers.Lambda(create_dist)],
      input_spec=observation_tensor_spec,
      name='actor_network')


def create_value_net(observation_tensor_spec: types.NestedTensorSpec,
                     seed=None) -> sequential.Sequential:
  """Create the value network."""
  seed_stream = tfp.util.SeedStream(seed=seed, salt='weight_init_seed')

  init = tf.keras.initializers.GlorotUniform(seed=seed_stream())

  dense = functools.partial(
      tf.keras.layers.Dense, activation='relu', kernel_initializer=init)
  fc_layer_units = [64, 64, 64, 64]

  def value_layer():
    return tf.keras.layers.Dense(
        1, activation=None, kernel_initializer=init, name='value')

  def drop_mask(observation_and_mask):
    return observation_and_mask['graph_embedding']

  def squeeze_value_dim(value):
    # Make value_prediction's shape from [B, T, 1] to [B, T].
    return tf.squeeze(value, -1, name='squeeze_value_net')

  return sequential.Sequential(
      [tf.keras.layers.Lambda(drop_mask)] + [tf.keras.layers.Flatten()] +
      [dense(num_units) for num_units in fc_layer_units] +
      [value_layer(), tf.keras.layers.Lambda(squeeze_value_dim)],
      input_spec=observation_tensor_spec,
      name='value_network')
