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
"""Sample training with distributed collection using a variable container."""

import functools
import os
import random
from typing import Dict, Optional

from absl import app
from absl import flags
from absl import logging
from circuit_training.environment import environment
from circuit_training.learning import static_feature_cache
from circuit_training.learning import train_ppo_lib
from circuit_training.model import create_models_lib
import gin
import numpy as np
import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils

_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', None, 'Paths to the gin-config files.'
)
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [], 'Gin binding parameters.'
)
_NETLIST_FILE = flags.DEFINE_multi_string(
    'netlist_file', None, 'File path to the netlist files.'
)
_INIT_PLACEMENT = flags.DEFINE_multi_string(
    'init_placement', None, 'File path to the init placement files.'
)
_STD_CELL_PLACER_MODE = flags.DEFINE_string(
    'std_cell_placer_mode',
    'dreamplace',
    (
        'Options for fast std cells placement: `fd` (uses the '
        'force-directed algorithm), `dreamplace` (uses DREAMPlace '
        'algorithm).'
    ),
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_REPLAY_BUFFER_SERVER_ADDR = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_VARIABLE_CONTAINER_SERVER_ADDR = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
)
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length',
    134,
    (
        'The sequence length to estimate shuffle size. Depends on the'
        ' environment.Max horizon = T translates to sequence_length T+1 because'
        ' of the additional boundary step (last -> first).'
    ),
)
_GLOBAL_SEED = flags.DEFINE_integer(
    'global_seed',
    111,
    'Used in env and weight initialization, does not impact action sampling.',
)
_POLICY_SAVED_MODEL_DIR = flags.DEFINE_string(
    'policy_saved_model_dir', None, 'If set, load the pretrained policy model.'
)
_POLICY_CHECKPOINT_DIR = flags.DEFINE_string(
    'policy_checkpoint_dir', None, 'If set, load the pretrained policy model.'
)

FLAGS = flags.FLAGS


def remove_tpu_in_the_name(var_name: str) -> str:
  """Removes the tpu in the tensor name."""
  return var_name.replace('_tpu_model', '_model')


def read_policy(
    policy_saved_model_dir: str, policy_checkpoint_dir: str
) -> Dict[str, tf.Tensor]:
  """Reads policy checkpoints."""
  checkpoints_prefix = os.path.join(
      policy_checkpoint_dir, 'variables', 'variables'
  )

  policy = tf.saved_model.load(policy_saved_model_dir)
  checkpoint = tf.train.Checkpoint(policy=policy)

  # assert_consumed to make sure all the variables are properly assgined.
  checkpoint.read(checkpoints_prefix).assert_consumed()

  policy_variable_map = {
      remove_tpu_in_the_name(v.name): v for v in policy.model_variables
  }
  return policy_variable_map


def load_policy_checkpoints(
    policy_saved_model_dir: str,
    policy_checkpoint_dir: str,
    actor_net: tf.keras.layers.Layer,
    value_net: tf.keras.layers.Layer,
) -> None:
  """Loads policy saved model.

  The weights will be loaded inplace.

  Args:
    policy_saved_model_dir: Policy saved model dir. It should contain at least
      variables and saved_model.pb.
    policy_checkpoint_dir: Policy checkpoint dir. Assumes the policy checkpoitns
      are in the format of policy_dir/variables/variables.index and
      policy_dir/variables/variables.data@1.
    actor_net: Actor network.
    value_net: Value network.

  Raises:
    If actor_net and value_net don't share the same network.
  """
  # All network variables are in policy_model. The value head
  # wraps the value network (part of the shared network) and does not contain
  # additional variables.
  actor_net_ref = {v.ref() for v in actor_net.variables}
  for v in value_net.variables:
    if v.ref() not in actor_net_ref:
      raise ValueError(
          'Currently we only support actor_net and value_net share the '
          'same variables.'
      )

  policy_variable_map = read_policy(
      policy_saved_model_dir, policy_checkpoint_dir
  )

  used_map = set()
  for variable in actor_net.variables:
    key = remove_tpu_in_the_name(variable.name)
    if key not in policy_variable_map:
      continue
    variable.assign(policy_variable_map[key])
    used_map.add(key)

  for key in policy_variable_map:
    assert key in used_map, f'Key {key} unused!'


def get_train_step_from_checkpoint(checkpoint_path: str) -> int:
  """Gets the train step from the checkpoint.

  Args:
    checkpoint_path: The path to the checkpoint. The path is expected to be:
      /path/to/policy_checkpoint_000111. In this case, 111 will be converted to
      train_step.

  Returns:
    The parsed train_step.

  Raises:
    If the train_step cannot be parsed.
  """
  train_step = os.path.basename(checkpoint_path).split('_')[-1]
  if train_step.isdigit():
    return int(train_step)
  else:
    raise ValueError(f'{train_step} cannot be converted to train_step.')


def try_load_checkpoint(
    root_dir: str,
    actor_net: tf.keras.layers.Layer,
    value_net: tf.keras.layers.Layer,
    policy_saved_model_dir: Optional[str] = None,
    policy_checkpoint_dir: Optional[str] = None,
):
  # Default init train step. It will be overwritten when loading from the
  # root_dir.
  init_train_step = 0

  current_saved_model_dir = os.path.join(
      root_dir, learner.POLICY_SAVED_MODEL_DIR
  )
  current_raw_policy_saved_model_dir = os.path.join(
      current_saved_model_dir, learner.RAW_POLICY_SAVED_MODEL_DIR
  )
  current_checkpoint_list = tf.io.gfile.glob(
      os.path.join(
          current_saved_model_dir,
          learner.POLICY_CHECKPOINT_DIR,
          'policy_checkpoint_*',
      )
  )

  load_from_root_dir = False
  # Resumes training in case of job restarts.
  if (
      tf.io.gfile.exists(current_saved_model_dir)
      and tf.io.gfile.exists(current_raw_policy_saved_model_dir)
      and current_checkpoint_list
  ):
    load_from_root_dir = True

    # Sorts the checkpoint_list in descending order.
    current_checkpoint_list = sorted(current_checkpoint_list, reverse=True)
    policy_checkpoint_latest_checkpoint = None
    for checkpoint in current_checkpoint_list:
      # Check the variable file in case it creates variable temp files.
      if not tf.io.gfile.exists(
          os.path.join(checkpoint, 'variables/variables.index')
      ):
        continue
      policy_checkpoint_latest_checkpoint = checkpoint
      break

    if policy_checkpoint_latest_checkpoint is None:
      logging.info(
          'Valid checkpoint cannot be found in the %s.', current_saved_model_dir
      )
      load_from_root_dir = False
    else:
      logging.info(
          'Loading initial checkpoint from latest checkpoint at %s.',
          policy_checkpoint_latest_checkpoint,
      )
      load_policy_checkpoints(
          current_raw_policy_saved_model_dir,
          policy_checkpoint_latest_checkpoint,
          actor_net,
          value_net,
      )

      try:
        init_train_step = get_train_step_from_checkpoint(
            policy_checkpoint_latest_checkpoint
        )
      except ValueError as e:
        # init_train_step will be set to 0 by default.
        logging.info(
            'Failed to get train_step from %s due to %s',
            policy_checkpoint_latest_checkpoint,
            e,
        )

  # If not loads from root_dir, loads from the input arguments.
  if not load_from_root_dir:
    if bool(policy_saved_model_dir) != bool(policy_checkpoint_dir):
      raise ValueError(
          f'Please make sure policy_saved_model_dir: {policy_saved_model_dir} '
          f'and policy_checkpoint_dir: {policy_checkpoint_dir} are both set or '
          'unset.'
      )

    # Loading from a pre-trained checkpoint.
    if policy_saved_model_dir:
      logging.info(
          'Loading initial checkpoint from canonical GRL at %s',
          policy_checkpoint_dir,
      )
      load_policy_checkpoints(
          policy_saved_model_dir, policy_checkpoint_dir, actor_net, value_net
      )

  return init_train_step


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILE.value, _GIN_BINDINGS.value, skip_unknown=True
  )

  logging.info('global seed=%d', _GLOBAL_SEED.value)
  np.random.seed(_GLOBAL_SEED.value)
  random.seed(_GLOBAL_SEED.value)
  tf.random.set_seed(_GLOBAL_SEED.value)

  root_dir = os.path.join(_ROOT_DIR.value, str(_GLOBAL_SEED.value))

  strategy = strategy_utils.get_strategy(
      strategy_utils.TPU.value, strategy_utils.USE_GPU.value
  )
  use_model_tpu = bool(strategy_utils.TPU.value)

  cache = static_feature_cache.StaticFeatureCache()

  assert len(_NETLIST_FILE.value) == len(
      _INIT_PLACEMENT.value
  ), 'Number of netlist and init files should be the same.'

  for netlist_index, (netlist_file, init_placement) in enumerate(
      zip(_NETLIST_FILE.value, _INIT_PLACEMENT.value)
  ):
    create_env_fn = functools.partial(
        environment.create_circuit_environment,
        netlist_file=netlist_file,
        init_placement=init_placement,
        global_seed=_GLOBAL_SEED.value,
        std_cell_placer_mode=_STD_CELL_PLACER_MODE.value,
        netlist_index=netlist_index,
    )
    env = create_env_fn()
    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(env)
    )
    static_features = env.wrapped_env().get_static_obs()
    cache.add_static_feature(static_features)

  with strategy.scope():
    actor_net, value_net = create_models_lib.create_models_fn(
        rl_architecture='generalization',
        observation_tensor_spec=observation_tensor_spec,
        action_tensor_spec=action_tensor_spec,
        static_features=cache.get_all_static_features(),
        use_model_tpu=use_model_tpu,
        seed=_GLOBAL_SEED.value,
    )

    actor_net.create_variables(training=False)
    value_net.create_variables(training=False)

    init_train_step = try_load_checkpoint(
        root_dir,
        actor_net,
        value_net,
        _POLICY_SAVED_MODEL_DIR.value,
        _POLICY_CHECKPOINT_DIR.value,
    )

  train_ppo_lib.train(
      root_dir=root_dir,
      strategy=strategy,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDR.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDR.value,
      action_tensor_spec=action_tensor_spec,
      time_step_tensor_spec=time_step_tensor_spec,
      sequence_length=_SEQUENCE_LENGTH.value,
      actor_net=actor_net,
      value_net=value_net,
      init_train_step=init_train_step,
      num_netlists=len(_NETLIST_FILE.value),
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'replay_buffer_server_address',
      'variable_container_server_address',
  ])
  multiprocessing.handle_main(functools.partial(app.run, main))
