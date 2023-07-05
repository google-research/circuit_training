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
"""circuit training agent definition and utility functions."""

from typing import Optional, Text, Tuple

from absl import logging
import gin
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity
from tf_agents.utils import value_ops


def _normalize_advantages(advantages, axes=(0), variance_epsilon=1e-8):
  adv_mean, adv_var = tf.nn.moments(x=advantages, axes=axes, keepdims=True)
  normalized_advantages = (advantages - adv_mean) / (
      tf.sqrt(adv_var) + variance_epsilon
  )
  return normalized_advantages


@gin.configurable
class CircuitPPOAgent(ppo_agent.PPOAgent):
  """A PPO Agent for circuit training aligned with Menger.

  Major differencs between this and ppo_agent.PPOAgent:
  - Loss aggregation uses reduce_mean instead of common.aggregate_losses which
    handles aggregation across multiple accelerator cores.
  - Value bootstrapping uses the second to last observation, instead of the
    last one. This is likely temporarily for aligning with Menger.
  - The additional time dimension ([B, 1, ...] was squeezed at the beginning,
    which eventually leads to different behavior when generating the action
    distribution. b/202055908 tracks the work on fully understanding and
    documenting this.
  - Normalization is done manually as opposed to `tf.nn.batch_normalization`
    which leads to different results in TPU setups.
  """

  def __init__(
      self,
      time_step_spec: ts.TimeStep,
      action_spec: types.NestedTensorSpec,
      optimizer: Optional[types.Optimizer] = None,
      actor_net: Optional[network.Network] = None,
      value_net: Optional[network.Network] = None,
      importance_ratio_clipping: types.Float = 0.2,
      discount_factor: types.Float = 1.0,
      entropy_regularization: types.Float = 0.01,
      value_pred_loss_coef: types.Float = 0.5,
      gradient_clipping: Optional[types.Float] = 1.0,
      value_clipping: Optional[types.Float] = None,
      check_numerics: bool = False,
      debug_summaries: bool = False,
      summarize_grads_and_vars: bool = False,
      train_step_counter: Optional[tf.Variable] = None,
      aggregate_losses_across_replicas: bool = True,
      report_loss_scaling_factor: float = 1.0,
      value_warmup_steps: int = 0,
      name: Optional[Text] = 'PPOClipAgent',
  ):
    """Creates a PPO Agent implementing the clipped probability ratios.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      optimizer: Optimizer to use for the agent.
      actor_net: A function actor_net(observations, action_spec) that returns
        tensor of action distribution params for each observation. Takes nested
        observation and returns nested action.
      value_net: A function value_net(time_steps) that returns value tensor from
        neural net predictions for each observation. Takes nested observation
        and returns batch of value_preds.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      discount_factor: Discount factor for return computation.
      entropy_regularization: Coefficient for entropy regularization loss term.
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      value_clipping: Difference between new and old value predictions are
        clipped to this threshold. Value clipping could be helpful when training
        very deep networks. Default: no clipping.
      check_numerics: If true, adds tf.debugging.check_numerics to help find NaN
        / Inf values. For debugging only.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      aggregate_losses_across_replicas: only applicable to setups using multiple
        relicas. Default to aggregating across multiple cores using common.
        aggregate_losses. If set to `False`, use `reduce_mean` directly, which
        is faster but may impact learning results.
      report_loss_scaling_factor: the multiplier for scaling the loss in the
        report.
      value_warmup_steps: How many learner steps (minibatches) to only backprop
        the value loss through the value head.
      name: The name of this agent. All variables in this module will fall under
        that name. Defaults to the class name.

    Raises:
      ValueError: If the actor_net is not a DistributionNetwork.
    """
    self._report_loss_scaling_factor = report_loss_scaling_factor
    self._use_tpu = bool(tf.config.list_logical_devices('TPU'))

    super(CircuitPPOAgent, self).__init__(
        time_step_spec,
        action_spec,
        optimizer,
        actor_net,
        value_net,
        importance_ratio_clipping=importance_ratio_clipping,
        discount_factor=discount_factor,
        entropy_regularization=entropy_regularization,
        value_pred_loss_coef=value_pred_loss_coef,
        gradient_clipping=gradient_clipping,
        value_clipping=value_clipping,
        check_numerics=check_numerics,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter,
        aggregate_losses_across_replicas=aggregate_losses_across_replicas,
        # Epochs are set through the tf.Data pipeline outside of the agent.
        num_epochs=1,
        # Value and advantages are computed as part of the data pipeline, this
        # is set to False for all setups using minibatching and PPOLearner.
        compute_value_and_advantage_in_train=False,
        normalize_rewards=False,
        normalize_observations=False,
        update_normalizers_in_train=False,
        name=name,
    )
    self._value_warmup_steps = value_warmup_steps

  def compute_return_and_advantage(
      self, next_time_steps: ts.TimeStep, value_preds: types.Tensor
  ) -> Tuple[types.Tensor, types.Tensor]:
    """Compute the Monte Carlo return and advantage.

    Args:
      next_time_steps: batched tensor of TimeStep tuples after action is taken.
      value_preds: Batched value prediction tensor. Should have one more entry
        in time index than time_steps, with the final value corresponding to the
        value prediction of the final state.

    Returns:
      tuple of (return, advantage), both are batched tensors.
    """
    discounts = next_time_steps.discount * tf.constant(
        self._discount_factor, dtype=tf.float32
    )

    rewards = next_time_steps.reward
    # TODO(b/202226773): Move debugging to helper function for clarity.
    if self._debug_summaries:
      # Summarize rewards before they get normalized below.
      # TODO(b/171573175): remove the condition once histograms are
      # supported on TPUs.
      if not self._use_tpu:
        tf.compat.v2.summary.histogram(
            name='rewards', data=rewards, step=self.train_step_counter
        )
      tf.compat.v2.summary.scalar(
          name='rewards_mean',
          data=tf.reduce_mean(rewards),
          step=self.train_step_counter,
      )

    # Normalize rewards if self._reward_normalizer is defined.
    if self._reward_normalizer:
      rewards = self._reward_normalizer.normalize(
          rewards, center_mean=False, clip_value=self._reward_norm_clipping
      )
      if self._debug_summaries:
        # TODO(b/171573175): remove the condition once histograms are
        # supported on TPUs.
        if not self._use_tpu:
          tf.compat.v2.summary.histogram(
              name='rewards_normalized',
              data=rewards,
              step=self.train_step_counter,
          )
        tf.compat.v2.summary.scalar(
            name='rewards_normalized_mean',
            data=tf.reduce_mean(rewards),
            step=self.train_step_counter,
        )

    # Make discount 0.0 at end of each episode to restart cumulative sum
    #   end of each episode.
    episode_mask = common.get_episode_mask(next_time_steps)
    discounts *= episode_mask

    # Compute Monte Carlo returns. Data from incomplete trajectories, not
    #   containing the end of an episode will also be used, with a bootstrapped
    #   estimation from the last value.
    # Note that when a trajectory driver is used, then the final step is
    #   terminal, the bootstrapped estimation will not be used, as it will be
    #   multiplied by zero (the discount on the last step).
    # TODO(b/202055908): Use -1 instead to bootstrap from the last step, once
    # we verify that it has no negative impact on learning.
    final_value_bootstrapped = value_preds[:, -2]
    returns = value_ops.discounted_return(
        rewards,
        discounts,
        time_major=False,
        final_value=final_value_bootstrapped,
    )
    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._debug_summaries and not self._use_tpu:
      tf.compat.v2.summary.histogram(
          name='returns', data=returns, step=self.train_step_counter
      )

    # Compute advantages.
    advantages = self.compute_advantages(
        rewards, returns, discounts, value_preds
    )

    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._debug_summaries and not self._use_tpu:
      tf.compat.v2.summary.histogram(
          name='advantages', data=advantages, step=self.train_step_counter
      )

    # Return TD-Lambda returns if both use_td_lambda_return and use_gae.
    if self._use_td_lambda_return:
      if not self._use_gae:
        logging.warning(
            'use_td_lambda_return was True, but use_gae was '
            'False. Using Monte Carlo return.'
        )
      else:
        returns = tf.add(
            advantages, value_preds[:, :-1], name='td_lambda_returns'
        )

    return returns, advantages

  def get_loss(
      self,
      time_steps: ts.TimeStep,
      actions: types.NestedTensorSpec,
      act_log_probs: types.Tensor,
      returns: types.Tensor,
      normalized_advantages: types.Tensor,
      action_distribution_parameters: types.NestedTensor,
      weights: types.Tensor,
      train_step: tf.Variable,
      debug_summaries: bool,
      old_value_predictions: Optional[types.Tensor] = None,
      finetune_value_only: bool = False,
      training: bool = False,
  ) -> tf_agent.LossInfo:
    """Compute the loss and create optimization op for one training epoch.

    All tensors should have a single batch dimension.

    Makes the following additions to PPOAgent:
      1. Adds option to only backprop through value head.

    Args:
      time_steps: A minibatch of TimeStep tuples.
      actions: A minibatch of actions.
      act_log_probs: A minibatch of action probabilities (probability under the
        sampling policy).
      returns: A minibatch of per-timestep returns.
      normalized_advantages: A minibatch of normalized per-timestep advantages.
      action_distribution_parameters: Parameters of data-collecting action
        distribution. Needed for KL computation.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      train_step: A train_step variable to increment for each train step.
        Typically the global_step.
      debug_summaries: True if debug summaries should be created.
      old_value_predictions: (Optional) The saved value predictions, used for
        calculating the value estimation loss when value clipping is performed.
      finetune_value_only: Whether to only make the value head trainable.
      training: Whether this loss is being used for training.

    Returns:
      A tf_agent.LossInfo named tuple with the total_loss and all intermediate
        losses in the extra field contained in a PPOLossInfo named tuple.
    """
    ppo_loss = super(CircuitPPOAgent, self).get_loss(
        time_steps,
        actions,
        act_log_probs,
        returns,
        normalized_advantages,
        action_distribution_parameters,
        weights,
        train_step,
        debug_summaries,
        old_value_predictions,
        training,
    )

    total_loss = tf.cond(
        finetune_value_only,
        lambda: ppo_loss.extra.value_estimation_loss,
        lambda: ppo_loss.loss,
    )

    # Properly report losses.
    policy_gradient_loss = tf.cond(
        finetune_value_only,
        lambda: tf.zeros_like(ppo_loss.extra.policy_gradient_loss),
        lambda: ppo_loss.extra.policy_gradient_loss,
    )
    entropy_regularization_loss = tf.cond(
        finetune_value_only,
        lambda: tf.zeros_like(ppo_loss.extra.entropy_regularization_loss),
        lambda: ppo_loss.extra.entropy_regularization_loss,
    )
    kl_penalty_loss = tf.cond(
        finetune_value_only,
        lambda: tf.zeros_like(ppo_loss.extra.kl_penalty_loss),
        lambda: ppo_loss.extra.kl_penalty_loss,
    )

    return tf_agent.LossInfo(
        total_loss,
        ppo_agent.PPOLossInfo(
            policy_gradient_loss=policy_gradient_loss,
            value_estimation_loss=ppo_loss.extra.value_estimation_loss,
            l2_regularization_loss=ppo_loss.extra.l2_regularization_loss,
            entropy_regularization_loss=entropy_regularization_loss,
            kl_penalty_loss=kl_penalty_loss,
            clip_fraction=self._clip_fraction,
        ),
    )

  def _train(self, experience, weights):
    experience = self._as_trajectory(experience)

    if self._compute_value_and_advantage_in_train:
      processed_experience = self._preprocess(experience)
    else:
      processed_experience = experience

    def squeeze_time_dim(t):
      return tf.squeeze(t, axis=[1])

    processed_experience = tf.nest.map_structure(
        squeeze_time_dim, processed_experience
    )

    valid_mask = ppo_utils.make_trajectory_mask(processed_experience)

    masked_weights = valid_mask
    if weights is not None:
      masked_weights *= weights

    # Reconstruct per-timestep policy distribution from stored distribution
    #   parameters.
    old_action_distribution_parameters = processed_experience.policy_info[
        'dist_params'
    ]

    old_actions_distribution = ppo_utils.distribution_from_spec(
        self._action_distribution_spec,
        old_action_distribution_parameters,
        legacy_distribution_network=isinstance(
            self._actor_net, network.DistributionNetwork
        ),
    )

    # Compute log probability of actions taken during data collection, using the
    #   collect policy distribution.
    old_act_log_probs = common.log_probability(
        old_actions_distribution, processed_experience.action, self._action_spec
    )

    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._debug_summaries and not self._use_tpu:
      actions_list = tf.nest.flatten(processed_experience.action)
      show_action_index = len(actions_list) != 1
      for i, single_action in enumerate(actions_list):
        action_name = 'actions_{}'.format(i) if show_action_index else 'actions'
        tf.compat.v2.summary.histogram(
            name=action_name, data=single_action, step=self.train_step_counter
        )

    time_steps = ts.TimeStep(
        step_type=processed_experience.step_type,
        reward=processed_experience.reward,
        discount=processed_experience.discount,
        observation=processed_experience.observation,
    )

    actions = processed_experience.action
    returns = processed_experience.policy_info['return']
    advantages = processed_experience.policy_info['advantage']

    normalized_advantages = _normalize_advantages(
        advantages, variance_epsilon=1e-8
    )

    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._debug_summaries and not self._use_tpu:
      tf.compat.v2.summary.histogram(
          name='advantages_normalized',
          data=normalized_advantages,
          step=self.train_step_counter,
      )
    old_value_predictions = processed_experience.policy_info['value_prediction']

    batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]

    loss_info = None  # TODO(b/123627451): Remove.
    variables_to_train = list(
        object_identity.ObjectIdentitySet(
            self._actor_net.trainable_weights
            + self._value_net.trainable_weights
        )
    )
    # Sort to ensure tensors on different processes end up in same order.
    variables_to_train = sorted(variables_to_train, key=lambda x: x.name)

    # Set finetune value only state in networks.
    finetune_value_only = self.train_step_counter < self._value_warmup_steps
    if self._value_warmup_steps:
      self._collect_policy._value_network.set_finetune_value_only(  # pylint: disable=protected-access
          finetune_value_only
      )

    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(variables_to_train)
      loss_info = self.get_loss(
          time_steps,
          actions,
          old_act_log_probs,
          returns,
          normalized_advantages,
          old_action_distribution_parameters,
          masked_weights,
          self.train_step_counter,
          self._debug_summaries,
          old_value_predictions=old_value_predictions,
          finetune_value_only=finetune_value_only,
          training=True,
      )

    grads = tape.gradient(loss_info.loss, variables_to_train)
    if self._gradient_clipping > 0:
      # global_norm may overflow or underflow. Clip and input it to the
      # clip_by_global_norm function to ensure numerical stability.
      global_norm = tf.linalg.global_norm(grads)
      global_norm = tf.clip_by_value(
          global_norm, clip_value_min=1e-25, clip_value_max=1e25
      )

      grads, _ = tf.clip_by_global_norm(
          grads, self._gradient_clipping, global_norm
      )

    self._grad_norm = tf.linalg.global_norm(grads)

    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(grads, variables_to_train))

    # If summarize_gradients, create functions for summarizing both
    # gradients and variables.
    if self._summarize_grads_and_vars and self._debug_summaries:
      eager_utils.add_gradients_summaries(
          grads_and_vars, self.train_step_counter
      )
      eager_utils.add_variables_summaries(
          grads_and_vars, self.train_step_counter
      )

    self._optimizer.apply_gradients(grads_and_vars)
    self.train_step_counter.assign_add(1)

    # TODO(b/161365079): Move this logic to PPOKLPenaltyAgent.
    if self._initial_adaptive_kl_beta > 0:
      # After update epochs, update adaptive kl beta, then update observation
      #   normalizer and reward normalizer.
      policy_state = self._collect_policy.get_initial_state(batch_size)
      # Compute the mean kl from previous action distribution.
      kl_divergence = self._kl_divergence(
          time_steps,
          old_action_distribution_parameters,
          self._collect_policy.distribution(time_steps, policy_state).action,
      )
      kl_divergence *= masked_weights
      self.update_adaptive_kl_beta(kl_divergence)

    if self.update_normalizers_in_train:
      self.update_observation_normalizer(time_steps.observation)
      self.update_reward_normalizer(processed_experience.reward)

    loss_info = tf.nest.map_structure(tf.identity, loss_info)

    with tf.name_scope('Losses/'):
      tf.compat.v2.summary.scalar(
          name='policy_gradient_loss',
          data=loss_info.extra.policy_gradient_loss
          * self._report_loss_scaling_factor,
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name='value_estimation_loss',
          data=loss_info.extra.value_estimation_loss
          * self._report_loss_scaling_factor,
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name='l2_regularization_loss',
          data=loss_info.extra.l2_regularization_loss
          * self._report_loss_scaling_factor,
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name='entropy_regularization_loss',
          data=loss_info.extra.entropy_regularization_loss
          * self._report_loss_scaling_factor,
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name='kl_penalty_loss',
          data=loss_info.extra.kl_penalty_loss
          * self._report_loss_scaling_factor,
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name='clip_fraction',
          data=loss_info.extra.clip_fraction,
          step=self.train_step_counter,
      )
      tf.compat.v2.summary.scalar(
          name='grad_norm', data=self._grad_norm, step=self.train_step_counter
      )

      total_abs_loss = (
          tf.abs(loss_info.extra.policy_gradient_loss)
          + tf.abs(loss_info.extra.value_estimation_loss)
          + tf.abs(loss_info.extra.entropy_regularization_loss)
          + tf.abs(loss_info.extra.l2_regularization_loss)
          + tf.abs(loss_info.extra.kl_penalty_loss)
      )

      tf.compat.v2.summary.scalar(
          name='total_abs_loss',
          data=total_abs_loss * self._report_loss_scaling_factor,
          step=self.train_step_counter,
      )

      tf.compat.v2.summary.scalar(
          name='total_loss',
          data=loss_info.loss * self._report_loss_scaling_factor,
          step=self.train_step_counter,
      )

    with tf.name_scope('LearningRate/'):
      tf.compat.v2.summary.scalar(
          name='learning_rate',
          data=self._optimizer.learning_rate,
          step=self.train_step_counter,
      )

    # TODO(b/171573175): remove the condition once histograms are
    # supported on TPUs.
    if self._summarize_grads_and_vars and not self._use_tpu:
      with tf.name_scope('Variables/'):
        all_vars = (
            self._actor_net.trainable_weights
            + self._value_net.trainable_weights
        )
        for var in all_vars:
          tf.compat.v2.summary.histogram(
              name=var.name.replace(':', '_'),
              data=var,
              step=self.train_step_counter,
          )

    return loss_info


@gin.configurable(allowlist=['aggregate_losses_across_replicas'])
def create_circuit_ppo_agent(
    train_step: tf.Variable,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: types.TimeStep,
    actor_net: network.Network,
    value_net: network.Network,
    strategy: tf.distribute.Strategy,
    optimizer: Optional[types.Optimizer] = None,
    aggregate_losses_across_replicas: bool = True,
    **kwargs
) -> CircuitPPOAgent:
  """Creates a PPO agent."""

  if aggregate_losses_across_replicas:
    report_loss_scaling_factor = strategy.num_replicas_in_sync
  else:
    report_loss_scaling_factor = 1.0

  return CircuitPPOAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      optimizer=optimizer,
      actor_net=actor_net,
      value_net=value_net,
      train_step_counter=train_step,
      aggregate_losses_across_replicas=aggregate_losses_across_replicas,
      report_loss_scaling_factor=report_loss_scaling_factor,
      **kwargs
  )
