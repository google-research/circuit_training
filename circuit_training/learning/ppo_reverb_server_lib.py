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
"""Library and util functions for reverb server."""
import os

from absl import logging

import reverb
import tensorflow as tf

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common


def start_reverb_server(root_dir, replay_buffer_capacity, port):
  """todo."""
  collect_policy_saved_model_path = os.path.join(
      root_dir, learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  saved_model_pb_path = os.path.join(collect_policy_saved_model_path,
                                     'saved_model.pb')
  try:
    # Wait for the collect policy to be outputed by learner (timeout after 2
    # days), then load it.
    train_utils.wait_for_file(
        saved_model_pb_path, sleep_time_secs=2, num_retries=86400)
    collect_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        collect_policy_saved_model_path, load_specs_from_pbtxt=True)
  except TimeoutError as e:
    # If the collect policy does not become available during the wait time of
    # the call `wait_for_file`, that probably means the learner is not running.
    logging.error('Could not get the file %s. Exiting.', saved_model_pb_path)
    raise e

  # Create the signature for the variable container holding the policy weights.
  train_step = train_utils.create_train_step()
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
      variables)
  logging.info('Signature of variables: \n%s', variable_container_signature)

  # Create the signature for the replay buffer holding observed experience.
  replay_buffer_signature = tensor_spec.from_spec(
      collect_policy.collect_data_spec)
  replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
  logging.info('Signature of experience: \n%s', replay_buffer_signature)

  # Crete and start the replay buffer and variable container server.
  # TODO(b/159130813): Optionally turn the reverb server pieces into a library.
  server = reverb.Server(
      tables=[
          # The remover does not matter because we clear the table at the end
          # of each global step. We assume that the table is large enough to
          # contain the data collected from one step (otherwise some data will
          # be dropped).
          reverb.Table(  # Replay buffer storing experience for training.
              name='training_table',
              sampler=reverb.selectors.MaxHeap(),
              remover=reverb.selectors.MinHeap(),
              # Menger sets this to 8, but empirically 1 learns better
              # consistently.
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_buffer_capacity,
              max_times_sampled=1,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Variable container storing policy parameters.
              name=reverb_variable_container.DEFAULT_TABLE,
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=1,
              max_times_sampled=0,
              signature=variable_container_signature,
          ),
      ],
      port=port)
  server.wait()
