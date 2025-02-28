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
"""Circuit training Environmnet with gin config."""

import datetime
import math
import os
import time
from typing import Any, Callable, Optional, Protocol

from absl import logging
from circuit_training.dreamplace import dreamplace_core
from circuit_training.dreamplace import dreamplace_util
from circuit_training.environment import coordinate_descent_placer as cd_placer
from circuit_training.environment import observation_config
from circuit_training.environment import observation_extractor
from circuit_training.environment import placement_util
from circuit_training.environment import plc_client
import gin
import gym
import numpy as np
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers

ObsType = dict[str, np.ndarray]
InfoType = dict[str, float]


DREAMPLACE_RUNTIME = 'dreamplace_runtime'
TOTAL_EPISODE_RUNTIME = 'total_episode_runtime'


class InfeasibleActionError(ValueError):
  """An infeasible action were passed to the env."""

  def __init__(self, action, mask):
    """Initialize an infeasible action error.

    Args:
      action: Infeasible action that was performed.
      mask: The mask associated with the current observation. mask[action] is
        `0` for infeasible actions.
    """
    ValueError.__init__(self, action, mask)
    self.action = action
    self.mask = mask

  def __str__(self):
    return 'Infeasible action (%s) when the mask is (%s)' % (
        self.action,
        self.mask,
    )


COST_COMPONENTS = ['wirelength', 'congestion', 'density']


class CostInfoFunctionCallable(Protocol):

  def __call__(
      self,
      plc: plc_client.PlacementCost,
      done: bool,
      infeasible_state: bool = False,
  ) -> tuple[float, dict[str, float]]:
    ...


@gin.configurable
def cost_info_function(
    plc: plc_client.PlacementCost,
    done: bool,
    infeasible_state: bool = False,
    wirelength_weight: float = 1.0,
    density_weight: float = 1.0,
    congestion_weight: float = 0.5,
) -> tuple[float, dict[str, float]]:
  """Returns the RL cost and info.

  Args:
    plc: Placement cost object.
    done: Set if it is the terminal step.
    infeasible_state: Set if it is an infeasible state.
    wirelength_weight:  Weight of wirelength in the reward function.
    density_weight: Weight of density in the reward function.
    congestion_weight: Weight of congestion in the reward function used only for
      legalizing the placement in greedy std cell placer.

  Returns:
    The RL cost.

  Raises:
    ValueError: When the cost mode is not supported.

  Notes: we found the default congestion and density weights more stable.
  """
  del infeasible_state
  proxy_cost = 0.0
  info = {cost: -1.0 for cost in COST_COMPONENTS}

  if not done:
    return proxy_cost, info

  if wirelength_weight > 0.0:
    info['wirelength'] = plc.get_cost()
    proxy_cost += wirelength_weight * info['wirelength']

  if congestion_weight > 0.0:
    info['congestion'] = plc.get_congestion_cost()
    proxy_cost += congestion_weight * info['congestion']

  if density_weight > 0.0:
    info['density'] = plc.get_density_cost()
    proxy_cost += density_weight * info['density']

  return proxy_cost, info


@gin.configurable
class CircuitEnv(object):
  """Defines the CircuitEnv class."""

  INFEASIBLE_REWARD = -4.0

  def __init__(
      self,
      netlist_file: str = '',
      init_placement: str = '',
      create_placement_cost_fn: Callable[
          ..., plc_client.PlacementCost
      ] = placement_util.create_placement_cost,
      std_cell_placer_mode: str = 'fd',
      cost_info_fn: CostInfoFunctionCallable = cost_info_function,
      global_seed: int = 0,
      netlist_index: int = 0,
      save_placement: bool = False,
      save_best_cost: bool = True,
      output_plc_file: str = '',
      cd_finetune: bool = False,
      cd_plc_file: str = 'ppo_cd_placement.plc',
      train_step: Optional[tf.Variable] = None,
      output_all_features: bool = False,
      node_order: str = 'descending_size_macro_first',
      node_order_file: str = '',
      save_snapshot: bool = True,
      save_partial_placement: bool = False,
      mixed_size_dp_at_infeasible: bool = True,
      dp_target_density: float = 0.425,
      dp_regioning: bool | None = None,
  ):
    """Creates a CircuitEnv.

    Args:
      netlist_file: Path to the input netlist file.
      init_placement: Path to the input initial placement file, used to read
        grid and canas size.
      create_placement_cost_fn: A function that given the netlist and initial
        placement file create the placement_cost object.
      std_cell_placer_mode: Options for fast std cells placement: `fd` (uses the
        force-directed algorithm).
      cost_info_fn: The cost function that given the plc object returns the RL
        cost.
      global_seed: Global seed for initializing env features. This seed should
        be the same across actors.
      netlist_index: Netlist index in the model static features.
      save_placement: If set, save the final placement in output_dir.
      save_best_cost: Boolean, if set, saves the palcement if its cost is better
        than the previously saved palcement.
      output_plc_file: The path to save the final placement.
      cd_finetune: If True, runs coordinate descent to finetune macro
        orientations. Supposed to run in eval only, not training.
      cd_plc_file: Name of the CD fine-tuned plc file, the file will be save in
        the same dir as output_plc_file
      train_step: A tf.Variable indicating the training step, only used for
        saving plc files in the evaluation.
      output_all_features: If true, it outputs all the observation features.
        Otherwise, it only outputs the dynamic observations.
      node_order: The sequence order of nodes placed by RL.
      save_snapshot: If true, save the snapshot placement.
      save_partial_placement: If true, eval also saves the placement even if RL
        does not place all nodes when an episode is done.
      mixed_size_dp_at_infeasible: If true, run mixed size DP at infeasible
        states. Only effective when std_cell_placer_mode is 'dreamplace'.
      dp_target_density: Target density parameter in DREAMPlace.
      dp_regioning: If set, use for regioning in DREAMPlace, if not set, use
        regioning is set only if there are mutliple power domains.
    """
    self._global_seed = global_seed
    if not netlist_file:
      raise ValueError('netlist_file must be provided.')

    self.netlist_file = netlist_file
    self._std_cell_placer_mode = std_cell_placer_mode
    self._cost_info_fn = cost_info_fn
    self._save_placement = save_placement
    self._save_best_cost = save_best_cost
    self._output_plc_file = output_plc_file
    self._output_plc_dir = os.path.dirname(output_plc_file)
    self._cd_finetune = cd_finetune
    self._cd_plc_file = cd_plc_file
    self._train_step = train_step
    self._netlist_index = netlist_index
    self._output_all_features = output_all_features
    self._node_order = node_order
    self._plc = create_placement_cost_fn(
        netlist_file=netlist_file, init_placement=init_placement
    )
    self._save_snapshot = save_snapshot
    self._save_partial_placement = save_partial_placement
    self._mixed_size_dp_at_infeasible = mixed_size_dp_at_infeasible

    self._observation_config = observation_config.ObservationConfig()

    self._grid_cols, self._grid_rows = self._plc.get_grid_num_columns_rows()
    self._canvas_width, self._canvas_height = (
        self._plc.get_canvas_width_height()
    )

    self._hard_macro_indices = [
        m
        for m in self._plc.get_macro_indices()
        if not (self._plc.is_node_soft_macro(m) or self._plc.is_node_fixed(m))
    ]
    self._num_hard_macros = len(self._hard_macro_indices)
    logging.info('***Num node to place***:%s', self._num_hard_macros)

    if node_order_file:
      self._sorted_node_indices = placement_util.get_ordered_node_indices(
          mode='file',
          plc=self._plc,
          seed=self._global_seed,
          node_order_file=node_order_file,
      )
    else:
      self._sorted_node_indices = placement_util.get_ordered_node_indices(
          mode=self._node_order, plc=self._plc, seed=self._global_seed
      )

    # Generate a map from actual macro_index to its position in
    # self.macro_indices. Needed because node adjacency matrix is in the same
    # node order of plc.get_macro_indices.
    self._macro_index_to_pos = {}
    for i, macro_index in enumerate(self._plc.get_macro_indices()):
      self._macro_index_to_pos[macro_index] = i

    self._saved_cost = np.inf

    if self._std_cell_placer_mode == 'dreamplace':
      self._dreamplace = self.create_dreamplace(
          dp_target_density=dp_target_density,
          regioning=dp_regioning,
      )

      # Call dreamplace mixed-size before making ObservationExtractor, so we
      # use its placement as the initial location in the features.
      logging.info('Run DP mix-sized to initialize the locations.')
      # Making all macros movable for a mixed-size.
      self._dreamplace.placedb_plc.update_num_non_movable_macros(
          plc=self._plc, num_non_movable_macros=0
      )
      converged = self._dreamplace.place()
      self._dreamplace.placedb_plc.write_movable_locations_to_plc(self._plc)
      if not converged:
        logging.warning("Initial DREAMPlace mixed-size didn't converge.")

      self._dp_mixed_macro_locations = {
          m: self._plc.get_node_location(m)
          for m in self._sorted_node_indices[: self._num_hard_macros]
      }
    else:  # fd
      # Call fd mixed-size before making ObservationExtractor, so we
      # use its placement as the initial location in the features.
      placement_util.fd_placement_schedule(
          plc=self._plc,
          num_steps=(100, 100, 100),
          io_factor=1.0,
          move_distance_factors=(1.0, 2.0, 2.0),
          attract_factor=(1.0e2, 1.0e-3, 1.0e-5),
          repel_factor=(0.0, 1.0e5, 1.0e6),
          use_current_loc=False,
          move_macros=True,
      )

    self._observation_extractor = observation_extractor.ObservationExtractor(
        plc=self._plc,
        observation_config=self._observation_config,
        netlist_index=self._netlist_index,
    )
    self.reset()

  @property
  def observation_space(self) -> gym.spaces.Space:
    """Env Observation space."""
    if self._output_all_features:
      return self._observation_config.observation_space

    return self._observation_config.dynamic_observation_space

  @property
  def action_space(self) -> gym.spaces.Space:
    return gym.spaces.Discrete(self._observation_config.max_grid_size**2)

  @property
  def environment_name(self) -> str:
    return self.netlist_file

  @property
  def observation_config(self) -> observation_config.ObservationConfig:
    return self._observation_config

  @property
  def grid_cols(self) -> int:
    return self._grid_cols

  @property
  def grid_rows(self) -> int:
    return self._grid_rows

  @property
  def macro_names(self) -> list[str]:
    macro_ids = self._sorted_node_indices[: self._num_hard_macros]
    return [self._plc.get_node_name(m) for m in macro_ids]

  def create_dreamplace(
      self,
      dp_target_density: float,
      regioning: bool | None = None,
  ) -> dreamplace_core.SoftMacroPlacer:
    """Creates the SoftMacroPlacer."""
    canvas_width, canvas_height = self._plc.get_canvas_width_height()
    if regioning is None:
      regioning = self._plc.has_area_constraint()
    elif regioning:
      # Even if user set regioning to True, we still enable it only when there
      # are multiple power domains.
      regioning = self._plc.has_area_constraint()
    dreamplace_params = dreamplace_util.get_dreamplace_params(
        target_density=dp_target_density,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        regioning=regioning,
    )
    # Dreamplace requires that movable nodes appear first
    # and then fixed nodes.
    # Since the first node to be placed (becoming fixed) is the first node in
    # _sorted_node_indices, we reverse the order and send it to dreamplace.
    hard_macro_order = self._sorted_node_indices[: self._num_hard_macros]
    hard_macro_order = hard_macro_order[::-1]
    return dreamplace_core.SoftMacroPlacer(
        self._plc, dreamplace_params, hard_macro_order
    )

  def get_static_obs(self):
    """Get the static observation for the environment.

    Static observations are invariant across steps on the same netlist, such as
    netlist metadata and the adj graphs. This should only be used for
    generalized RL.

    Returns:
      Numpy array representing the observation
    """
    return self._observation_extractor.get_static_features()

  def get_cost_info(self, done: bool = False) -> tuple[float, dict[str, float]]:
    return self._cost_info_fn(plc=self._plc, done=done, infeasible_state=False)

  def _get_mask(self) -> np.ndarray:
    """Gets the node mask for the current node.

    Returns:
      List of 0s and 1s indicating if action is feasible or not.
    """
    if self._done:
      mask = np.zeros(self._observation_config.max_grid_size**2, dtype=np.int32)
    else:
      node_index = self._sorted_node_indices[self._current_node]
      mask = np.asarray(self._plc.get_node_mask(node_index), dtype=np.int32)
      mask = np.reshape(mask, [self._grid_rows, self._grid_cols])
      pad = (
          (
              self._observation_extractor.up_pad,
              self._observation_extractor.low_pad,
          ),
          (
              self._observation_extractor.right_pad,
              self._observation_extractor.left_pad,
          ),
      )
      mask = np.pad(mask, pad, mode='constant', constant_values=0)
    return np.reshape(
        mask, (self._observation_config.max_grid_size**2,)
    ).astype(np.int32)

  def _set_current_mask(self, mask: np.ndarray) -> None:
    self._current_mask = mask

  def _get_obs(self) -> ObsType:
    """Returns the observation."""
    if self._current_node > 0:
      previous_node_sorted = self._sorted_node_indices[self._current_node - 1]
      previous_node_index = self._macro_index_to_pos[previous_node_sorted]
    else:
      previous_node_index = -1

    if self._current_node < self._num_hard_macros:
      current_node_sorted = self._sorted_node_indices[self._current_node]
      current_node_index = self._macro_index_to_pos[current_node_sorted]
    else:
      current_node_index = 0

    if self._output_all_features:
      return self._observation_extractor.get_all_features(
          previous_node_index=previous_node_index,
          current_node_index=current_node_index,
          mask=self._current_mask,
      )
    else:
      return self._observation_extractor.get_dynamic_features(
          previous_node_index=previous_node_index,
          current_node_index=current_node_index,
          mask=self._current_mask,
      )

  def _run_cd(self):
    """Runs coordinate descent to finetune the current placement."""

    # CD only modifies macro orientation.
    # Plc modified by CD will be reset at the end of the episode.

    def cost_fn(plc):
      return self._cost_info_fn(plc=plc, done=True, infeasible_state=False)

    cd = cd_placer.CoordinateDescentPlacer(plc=self._plc, cost_fn=cost_fn)
    cd.place()

  def _save_placement_fn(self, cost: float) -> None:
    """Saves the current placement.

    Args:
      cost: the current placement cost.

    Raises:
      IOError: If we cannot write the placement to file.
    """
    if not self._save_best_cost or (
        cost < self._saved_cost
        and (math.fabs(cost - self._saved_cost) / (cost) > 5e-3)
    ):
      user_comments = ''
      if self._train_step:
        user_comments = f'Train step : {self._train_step.numpy()}'

      placement_util.save_placement(
          self._plc, self._output_plc_file, user_comments
      )
      self._saved_cost = cost

      ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
      if self._save_snapshot:
        ppo_snapshot_file = os.path.join(
            self._output_plc_dir,
            f'snapshot_ppo_opt_placement_timestamp_{ts}_cost_{cost:.4f}.plc',
        )
        placement_util.save_placement(
            self._plc, ppo_snapshot_file, user_comments
        )

      # Only runs CD if this is the best RL placement seen so far.
      if self._cd_finetune:
        self._run_cd()
        cost = self._cost_info_fn(
            plc=self._plc, done=True, infeasible_state=False
        )[0]
        cd_plc_file = os.path.join(self._output_plc_dir, self._cd_plc_file)
        placement_util.save_placement(self._plc, cd_plc_file, user_comments)
        cd_snapshot_file = os.path.join(
            self._output_plc_dir,
            f'snapshot_ppo_cd_placement_timestamp_{ts}_cost_{cost:.4f}.plc',
        )
        placement_util.save_placement(
            self._plc, cd_snapshot_file, user_comments
        )

  def call_analytical_placer_and_get_cost(
      self, infeasible_state=False
  ) -> tuple[float, InfoType]:
    """Calls analytical placer.

    Calls analystical placer and evaluates cost when all nodes are placed. Also,
    saves the placement file for eval if all the macros are placed by RL.

    Args:
      infeasible_state: If the function called for an infeasible state.

    Returns:
      A tuple for placement cost and info.
    """
    total_time = 0.0
    if self._done:
      start_time = time.time()
      self.analytical_placer()
      total_time = time.time() - start_time
    # Only evaluates placement cost when all nodes are placed.
    # All samples in the episode receive the same reward equal to final cost.
    # This is realized by setting intermediate steps cost as zero, and
    # propagate the final cost with discount factor set to 1 in replay buffer.
    cost, info = self._cost_info_fn(
        plc=self._plc, done=self._done, infeasible_state=infeasible_state
    )
    info[DREAMPLACE_RUNTIME] = total_time

    # Happens when the episode is done, when RL places all nodes, or we want to
    # save partial placement regardless RL places all nodes.
    if self._save_placement:
      if self._current_node == self._num_hard_macros or (
          self._done and self._save_partial_placement
      ):
        self._save_placement_fn(cost)

    info[TOTAL_EPISODE_RUNTIME] = time.time() - self._episode_start_time

    return -cost, info

  def reset(self) -> ObsType:
    """Restes the environment.

    Returns:
      An initial observation.
    """
    self._plc.unplace_all_nodes()
    self._current_actions = []
    self._current_node = 0
    self._done = False
    self._current_mask = self._get_mask()
    self._observation_extractor.reset()
    self._episode_start_time = time.time()
    return self._get_obs()

  def translate_to_original_canvas(self, action: int) -> int:
    """Translates a padded location to real one in the original canvas."""
    up_pad = (self._observation_config.max_grid_size - self._grid_rows) // 2
    right_pad = (self._observation_config.max_grid_size - self._grid_cols) // 2

    a_i = action // self._observation_config.max_grid_size - up_pad
    a_j = action % self._observation_config.max_grid_size - right_pad
    if 0 <= a_i < self._grid_rows or 0 <= a_j < self._grid_cols:
      action = a_i * self._grid_cols + a_j
    else:
      raise InfeasibleActionError(action, self._current_mask)
    return action

  def translate_to_padded_canvas(self, action: int) -> int:
    """Translates a real location to the padded one in the padded canvas."""
    up_pad = (self._observation_config.max_grid_size - self._grid_rows) // 2
    right_pad = (self._observation_config.max_grid_size - self._grid_cols) // 2

    if up_pad < 0 or right_pad < 0:
      raise ValueError(
          f'grid_rows {self._grid_rows} or grid_cols '
          f'{self._grid_cols} is larger than max_grid_size '
          f'{self._observation_config.max_grid_size}"'
      )

    a_i = action // self._grid_cols + up_pad
    a_j = action % self._grid_cols + right_pad

    return a_i * self._observation_config.max_grid_size + a_j

  def place_node(self, node_index: int, action: int) -> None:
    self._plc.place_node(node_index, self.translate_to_original_canvas(action))

  def analytical_placer(self) -> None:
    """Calls analytical placer to place stdcells or mix-size nodes."""
    if self._std_cell_placer_mode == 'dreamplace':
      self._dreamplace.placedb_plc.read_hard_macros_from_plc(self._plc)
      # We always update the placedb with number of placed macros, if the
      # previous number of fixed macros are the same as the current, the
      # expensive placedb conversion won't be called.
      self._dreamplace.placedb_plc.update_num_non_movable_macros(
          plc=self._plc, num_non_movable_macros=self._current_node
      )
      converged = self._dreamplace.place()
      if not converged:
        logging.warning("DREAMPlace didn't converge.")
      self._dreamplace.placedb_plc.write_movable_locations_to_plc(self._plc)
    elif self._std_cell_placer_mode == 'fd':
      placement_util.fd_placement_schedule(self._plc)
    else:
      raise ValueError(
          '%s is not a supported std_cell_placer_mode.'
          % (self._std_cell_placer_mode)
      )

  def step(self, action: int) -> tuple[ObsType, float, bool, Any]:
    """Steps the environment.

    Args:
      action: The action to take (should be a list of size 1).

    Returns:
      observation, reward, done, and info.

    Raises:
      RuntimeError: action taken after episode was done
      InfeasibleActionError: bad action taken (action is not in feasible
        actions)
    """
    if self._done:
      raise RuntimeError('Action taken after episode is done.')

    action = int(action)
    self._current_actions.append(action)
    if self._current_mask[action] == 0:
      raise InfeasibleActionError(action, self._current_mask)

    node_index = self._sorted_node_indices[self._current_node]
    self.place_node(node_index, action)

    self._current_node += 1
    self._done = self._current_node == self._num_hard_macros
    self._current_mask = self._get_mask()

    if not self._done and not np.any(self._current_mask):
      self._done = True
      logging.info(
          'Actions took before becoming infeasible: %s', self._current_actions
      )
      if (
          self._std_cell_placer_mode == 'dreamplace'
          and self._mixed_size_dp_at_infeasible
      ):
        logging.info(
            'Using DREAMPlace mixed-size placer for the rest of the macros and'
            ' std cell clusters.'
        )
        cost, info = self.call_analytical_placer_and_get_cost(
            infeasible_state=True
        )
        return self._get_obs(), cost, True, info
      else:
        info = {cost: -1.0 for cost in COST_COMPONENTS}
        return self._get_obs(), self.INFEASIBLE_REWARD, True, info

    cost, info = self.call_analytical_placer_and_get_cost()

    return self._get_obs(), cost, self._done, info


def create_circuit_environment(*args, **kwarg) -> wrappers.ActionClipWrapper:
  """Create an `CircuitEnv` wrapped as a Gym environment.

  Args:
    *args: Arguments.
    **kwarg: keyworded Arguments.

  Returns:
    PyEnvironment used for training.
  """
  env = CircuitEnv(*args, **kwarg)

  return wrappers.ActionClipWrapper(suite_gym.wrap_env(env))
