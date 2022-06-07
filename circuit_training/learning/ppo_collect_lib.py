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
"""Library for PPO collect job."""
import os

from absl import logging

import reverb

from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common


def collect(task,
            root_dir,
            replay_buffer_server_address,
            variable_container_server_address,
            create_env_fn,
            max_sequence_length,
            write_summaries_task_threshold=1):
  """Collects experience using a policy updated after every episode."""
  # Create the environment.
  train_step = train_utils.create_train_step()
  env = create_env_fn(train_step=train_step)

  # Create the path for the serialized collect policy.
  policy_saved_model_path = os.path.join(root_dir,
                                         learner.POLICY_SAVED_MODEL_DIR,
                                         learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  saved_model_pb_path = os.path.join(policy_saved_model_path, 'saved_model.pb')
  try:
    # Wait for the collect policy to be outputed by learner (timeout after 2
    # days), then load it.
    train_utils.wait_for_file(
        saved_model_pb_path, sleep_time_secs=2, num_retries=86400)
    policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        policy_saved_model_path, load_specs_from_pbtxt=True)
  except TimeoutError as e:
    # If the collect policy does not become available during the wait time of
    # the call `wait_for_file`, that probably means the learner is not running.
    logging.error('Could not get the file %s. Exiting.', saved_model_pb_path)
    raise e

  # Create the variable container.
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE])
  variable_container.update(variables)

  # Create the replay buffer observer for collect jobs.
  observers = [
      reverb_utils.ReverbAddEpisodeObserver(
          reverb.Client(replay_buffer_server_address),
          table_name=['training_table'],
          max_sequence_length=max_sequence_length,
          priority=model_id)
  ]

  # Write metrics only if the task ID of the current job is below the limit.
  summary_dir = None
  metrics = []
  if task < write_summaries_task_threshold:
    summary_dir = os.path.join(root_dir, learner.TRAIN_DIR, str(task))
    metrics = actor.collect_metrics(1)

  # Create the collect actor.
  collect_actor = actor.Actor(
      env,
      policy,
      train_step,
      episodes_per_run=1,
      summary_dir=summary_dir,
      metrics=metrics,
      observers=observers)

  # Run the experience collection loop.
  while True:
    collect_actor.run()
    variable_container.update(variables)
    logging.info('Collecting at step: %d', train_step.numpy())
    logging.info('Collecting at model_id: %d', model_id.numpy())
