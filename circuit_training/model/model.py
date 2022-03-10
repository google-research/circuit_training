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
"""Circtuittraining GRL Model."""

from typing import Optional, Text

from absl import logging
from circuit_training.model import model_lib
import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.specs import distribution_spec
import tf_agents.specs.tensor_spec as tensor_spec
from tf_agents.typing import types
from tf_agents.utils import nest_utils


@gin.configurable(module='circuittraining.models')
class GrlModel(network.Network):
  """Circuit GRL Model used as part of the canonical version."""

  def __init__(self,
               input_tensors_spec: types.NestedTensorSpec,
               output_tensors_spec: types.NestedTensorSpec,
               name: Optional[Text] = None,
               state_spec=(),
               policy_noise_weight: float = 0.0,
               static_features=None,
               use_model_tpu=True):

    super(GrlModel, self).__init__(
        input_tensor_spec=input_tensors_spec, state_spec=state_spec, name=name)

    if static_features:
      logging.info('Static features are passed to the model construction.')

    if use_model_tpu:
      self._model = model_lib.CircuitTrainingTPUModel(
          policy_noise_weight=policy_noise_weight,
          static_features=static_features)
    else:
      self._model = model_lib.CircuitTrainingModel(
          policy_noise_weight=policy_noise_weight,
          static_features=static_features)

  def call(self, inputs, network_state=()):
    logits, value = self._model(inputs)
    return {'logits': logits, 'value': value}, network_state


@gin.configurable(module='circuittraining.models')
class GrlPolicyModel(network.DistributionNetwork):
  """Circuit GRL Model."""

  def __init__(self, shared_network: network.Network,
               input_tensors_spec: types.NestedTensorSpec,
               output_tensors_spec: types.NestedTensorSpec,
               name: Optional[Text] = 'GrlPolicyModel'):

    super(GrlPolicyModel, self).__init__(
        input_tensor_spec=input_tensors_spec,
        state_spec=(),
        output_spec=output_tensors_spec,
        name=name)

    self._input_tensors_spec = input_tensors_spec
    self._shared_network = shared_network
    self._output_tensors_spec = output_tensors_spec

    n_unique_actions = np.unique(output_tensors_spec.maximum -
                                 output_tensors_spec.minimum + 1)
    input_param_spec = {
        'logits':
            tensor_spec.TensorSpec(
                shape=n_unique_actions,
                dtype=tf.float32,
                name=name + '_logits')
    }
    self._output_dist_spec = distribution_spec.DistributionSpec(
        tfp.distributions.Categorical,
        input_param_spec,
        sample_spec=output_tensors_spec,
        dtype=output_tensors_spec.dtype)

  @property
  def output_spec(self):
    return self._output_dist_spec

  @property
  def distribution_tensor_spec(self):
    return self._output_dist_spec

  def call(self, inputs, step_types=None, network_state=()):
    outer_rank = nest_utils.get_outer_rank(inputs, self._input_tensors_spec)
    if outer_rank == 0:
      inputs = tf.nest.map_structure(lambda x: tf.reshape(x, (1, -1)), inputs)
    model_out, _ = self._shared_network(inputs)

    paddings = tf.ones_like(inputs['mask'], dtype=tf.float32) * (-2.**32 + 1)
    masked_logits = tf.where(
        tf.cast(inputs['mask'], tf.bool), model_out['logits']['location'],
        paddings)

    output_dist = self._output_dist_spec.build_distribution(
        logits=masked_logits)

    return output_dist, network_state


@gin.configurable(module='circuittraining.models')
class GrlValueModel(network.Network):
  """Circuit GRL Model."""

  def __init__(self, input_tensors_spec: types.NestedTensorSpec,
               shared_network: network.Network, name: Optional[Text] = None):

    super(GrlValueModel, self).__init__(
        input_tensor_spec=input_tensors_spec, state_spec=(), name=name)

    self._input_tensors_spec = input_tensors_spec
    self._shared_network = shared_network

  def call(self, inputs, step_types=None, network_state=()):
    outer_rank = nest_utils.get_outer_rank(inputs,
                                           self._input_tensors_spec)
    if outer_rank == 0:
      inputs = tf.nest.map_structure(lambda x: tf.reshape(x, (1, -1)), inputs)
    model_out, _ = self._shared_network(inputs)

    def squeeze_value_dim(value):
      # Make value_prediction's shape from [B, T, 1] to [B, T].
      return tf.squeeze(value, -1)
    return squeeze_value_dim(model_out['value']), network_state


def create_grl_models(observation_tensor_spec,
                      action_tensor_spec,
                      static_features,
                      strategy,
                      use_model_tpu=False):
  """Create the GRL actor and value networks from scratch.

  Args:
    observation_tensor_spec: tensor spec for the observations.
    action_tensor_spec: tensor spec for the actions.
    static_features: static features from the environment to pass into the
      models. If None, read from the observations.
    strategy: the tf.distribute strategy to create the models under.
    use_model_tpu: boolean flag indicating the versions of the GRL models to
      create. TPU models leverage map_fn to speed up performance on TPUs. Both
      versions generate the same output given the same inputs.

  Returns:
    A tuple containing the GRL policy model and value model.

  """
  with strategy.scope():
    grl_shared_net = GrlModel(
        observation_tensor_spec,
        action_tensor_spec,
        static_features=static_features,
        use_model_tpu=use_model_tpu,
    )
    grl_actor_net = GrlPolicyModel(grl_shared_net, observation_tensor_spec,
                                   action_tensor_spec)
    grl_value_net = GrlValueModel(observation_tensor_spec, grl_shared_net)
    return grl_actor_net, grl_value_net
