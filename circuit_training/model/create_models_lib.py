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
"""Utils for creating models."""
from typing import Any, Dict

from circuit_training.model import fully_connected_model_lib
from circuit_training.model import model
import numpy as np
from tf_agents.typing import types


def create_models_fn(
    rl_architecture: str,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    static_features: Dict[str, np.ndarray],
    use_model_tpu: bool = False,
    seed: int = 0,
) -> tuple[Any, Any]:
  """Creates actor/value networks.

  Args:
    rl_architecture: The RL architecture.
    observation_tensor_spec: Env observation spec.
    action_tensor_spec: Env action spec.
    static_features: Env static features.
    use_model_tpu: Use TPU model.
    seed: Random seed.

  Returns:
    Tuple of actor_net, value_net.
  """
  if rl_architecture == 'generalization':
    actor_net, value_net = model.create_grl_models(
        observation_tensor_spec,
        action_tensor_spec,
        static_features,
        use_model_tpu=use_model_tpu,
        seed=seed,
    )
  else:
    actor_net = fully_connected_model_lib.create_actor_net(
        observation_tensor_spec, action_tensor_spec
    )
    value_net = fully_connected_model_lib.create_value_net(
        observation_tensor_spec
    )

  return actor_net, value_net
