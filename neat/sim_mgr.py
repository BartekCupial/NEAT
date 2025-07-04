import copy
import logging
import time
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from evojax.obs_norm import ObsNormalizer
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.sim_mgr import (
    all_done,
    create_logger,
    duplicate_params,
    get_task_reset_keys,
    merge_state_from_pmap,
    report_score,
    reshape_data_from_pmap,
    split_params_for_pmap,
    split_states_for_pmap,
    tree_map,
    update_score_and_mask,
)
from evojax.task.base import TaskState, VectorizedTask
from jax import random


def monkey_duplicate_params(
    params: Tuple[Dict, Dict], repeats: int, ma_training: bool, batch: bool = False
) -> Tuple[Dict, Dict]:
    """Enhanced duplicate_params that handles both arrays and dictionaries."""
    diff_params, static_params = params

    if batch:
        diff_params = {key: value[None, :] for key, value in diff_params.items()}
        static_params = {key: value[None, :] for key, value in static_params.items()}

    diff_params = {key: duplicate_params(value, repeats, ma_training) for key, value in diff_params.items()}
    static_params = {key: duplicate_params(value, repeats, ma_training) for key, value in static_params.items()}

    return diff_params, static_params


class BackpropSimManager(object):
    """Simulation manager."""

    def __init__(
        self,
        n_repeats: int,
        test_n_repeats: int,
        pop_size: int,
        n_evaluations: int,
        policy_net: PolicyNetwork,
        train_vec_task: VectorizedTask,
        valid_vec_task: VectorizedTask,
        seed: int = 0,
        obs_normalizer: ObsNormalizer = None,
        use_for_loop: bool = False,
        logger: logging.Logger = None,
        use_backprop: bool = True,
        backprop_steps: int = 10,
        learning_rate: float = 0.001,
        l2_penalty: float = 0.0,
        complexity_penalty: float = 0.0,
        optimizer: str = "adam",
    ):
        """Initialization function.

        Args:
            n_repeats - Number of repeated parameter evaluations.
            pop_size - Population size.
            n_evaluations - Number of evaluations of the best parameter.
            policy_net - Policy network.
            train_vec_task - Vectorized tasks for training.
            valid_vec_task - Vectorized tasks for validation.
            seed - Random seed.
            obs_normalizer - Observation normalization helper.
            use_for_loop - Use for loop for rollout instead of jax.lax.scan.
            logger - Logger.
            backprop_steps - Number of gradient descent steps to perform
            learning_rate - Learning rate for gradient descent
            l2_penalty - L2 penalty for regularization
            complexity_penalty - Complexity penalty for NEAT
            optimizer - Optimizer type ('adam', 'sgd', 'rmsprop')
        """

        if logger is None:
            self._logger = create_logger(name="SimManager")
        else:
            self._logger = logger

        self._use_for_loop = use_for_loop
        self._logger.info("use_for_loop={}".format(self._use_for_loop))
        self._key = random.PRNGKey(seed=seed)
        self._n_repeats = n_repeats
        self._test_n_repeats = test_n_repeats
        self._pop_size = pop_size
        self._n_evaluations = max(n_evaluations, jax.local_device_count())
        self._ma_training = train_vec_task.multi_agent_training

        self.obs_normalizer = obs_normalizer
        if self.obs_normalizer is None:
            self.obs_normalizer = ObsNormalizer(
                obs_shape=train_vec_task.obs_shape,
                dummy=True,
            )
        self.obs_params = self.obs_normalizer.get_init_params()

        self._num_device = jax.local_device_count()
        if self._pop_size % self._num_device != 0:
            raise ValueError(
                "pop_size must be multiples of GPU/TPUs: "
                "pop_size={}, #devices={}".format(self._pop_size, self._num_device)
            )
        if self._n_evaluations % self._num_device != 0:
            raise ValueError(
                "n_evaluations must be multiples of GPU/TPUs: "
                "n_evaluations={}, #devices={}".format(self._n_evaluations, self._num_device)
            )

        self._use_backprop = use_backprop
        self._backprop_steps = backprop_steps
        self._learning_rate = learning_rate
        self._l2_penalty = l2_penalty
        self._complexity_penalty = complexity_penalty

        # Initialize optimizer
        if optimizer == "adam":
            self._optimizer = optax.adam(learning_rate)
        elif optimizer == "sgd":
            self._optimizer = optax.sgd(learning_rate)
        elif optimizer == "rmsprop":
            self._optimizer = optax.rmsprop(learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        def step_once(carry, input_data, task):
            (task_state, policy_state, params, obs_params, accumulated_reward, valid_mask) = carry
            if task.multi_agent_training:
                num_tasks, num_agents = task_state.obs.shape[:2]
                task_state = task_state.replace(obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:])))
            org_obs = task_state.obs
            normed_obs = self.obs_normalizer.normalize_obs(org_obs, obs_params)
            task_state = task_state.replace(obs=normed_obs)
            actions, policy_state = policy_net.get_actions(task_state, params, policy_state)
            if task.multi_agent_training:
                task_state = task_state.replace(
                    obs=task_state.obs.reshape((num_tasks, num_agents, *task_state.obs.shape[1:]))
                )
                actions = actions.reshape((num_tasks, num_agents, *actions.shape[1:]))
            task_state, reward, done = task.step(task_state, actions)
            if task.multi_agent_training:
                reward = reward.ravel()
                done = jnp.repeat(done, num_agents, axis=0)
            accumulated_reward = accumulated_reward + reward * valid_mask
            valid_mask = valid_mask * (1 - done.ravel())
            return (
                (task_state, policy_state, params, obs_params, accumulated_reward, valid_mask),
                (org_obs, valid_mask),
            )

        def rollout(task_states, policy_states, params, obs_params, step_once_fn, max_steps):
            accumulated_rewards = jnp.zeros(params[0]["weights"].shape[0])
            valid_masks = jnp.ones(params[0]["weights"].shape[0])
            (
                (task_states, policy_states, params, obs_params, accumulated_rewards, valid_masks),
                (obs_set, obs_mask),
            ) = jax.lax.scan(
                step_once_fn,
                (task_states, policy_states, params, obs_params, accumulated_rewards, valid_masks),
                (),
                max_steps,
            )

            if self._complexity_penalty > 0.0:
                # Calculate the complexity penalty based on the number of parameters.
                diff_params, static_params = params
                weights = diff_params["weights"]
                # define complexity as the number of connections multiplied by the number of nodes
                complexity_per_network = jnp.sum(weights > 0, axis=range(1, weights.ndim)) * jnp.max(
                    static_params["output_indices"], axis=1
                )

                complexity_penalty = self._complexity_penalty * jnp.mean(complexity_per_network)
                accumulated_rewards -= complexity_penalty

            return accumulated_rewards, obs_set, obs_mask, task_states

        self._policy_reset_fn = jax.jit(policy_net.reset)
        self._policy_act_fn = jax.jit(policy_net.get_actions)

        if hasattr(train_vec_task, "bd_extractor") and train_vec_task.bd_extractor is not None:
            self._bd_summarize_fn = jax.jit(train_vec_task.bd_extractor.summarize)
        else:
            self._bd_summarize_fn = lambda x: x

        # Set up training functions.
        self._train_reset_fn = train_vec_task.reset
        self._train_step_fn = train_vec_task.step
        self._train_max_steps = train_vec_task.max_steps
        self._train_rollout_fn = partial(
            rollout, step_once_fn=partial(step_once, task=train_vec_task), max_steps=train_vec_task.max_steps
        )
        if self._num_device > 1:
            self._train_rollout_fn = jax.jit(jax.pmap(self._train_rollout_fn, in_axes=(0, 0, 0, None)))

        # Set up validation functions.
        self._valid_reset_fn = valid_vec_task.reset
        self._valid_step_fn = valid_vec_task.step
        self._valid_max_steps = valid_vec_task.max_steps
        self._valid_rollout_fn = partial(
            rollout, step_once_fn=partial(step_once, task=valid_vec_task), max_steps=valid_vec_task.max_steps
        )
        if self._num_device > 1:
            self._valid_rollout_fn = jax.jit(jax.pmap(self._valid_rollout_fn, in_axes=(0, 0, 0, None)))

    def eval_params(self, params: jnp.ndarray, test: bool) -> Tuple[jnp.ndarray, TaskState, Tuple[Dict, Dict]]:
        """Evaluate population parameters or test the best parameter.

        Args:
            params - Parameters to be evaluated.
            test - Whether we are testing the best parameter
        Returns:
            An array of fitness scores.
        """
        if self._use_for_loop:
            return self._for_loop_eval(params, test)
        else:
            return self._scan_loop_eval(params, test)

    def _for_loop_eval(self, params: jnp.ndarray, test: bool) -> Tuple[jnp.ndarray, TaskState, Tuple[Dict, Dict]]:
        """Rollout using for loop (no multi-device or ma_training yet)."""
        policy_reset_func = self._policy_reset_fn
        policy_act_func = self._policy_act_fn
        if test:
            n_repeats = self._test_n_repeats
            task_reset_func = self._valid_reset_fn
            task_step_func = self._valid_step_fn
            task_max_steps = self._valid_max_steps
            params = monkey_duplicate_params(params, self._n_evaluations, False, batch=True)
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
            task_step_func = self._train_step_fn
            task_max_steps = self._train_max_steps

        params = monkey_duplicate_params(params, n_repeats, self._ma_training)

        # Start rollout.
        self._key, reset_keys = get_task_reset_keys(
            self._key, test, self._pop_size, self._n_evaluations, n_repeats, self._ma_training
        )
        task_state = task_reset_func(reset_keys)
        policy_state = policy_reset_func(task_state)
        scores = jnp.zeros(params.shape[0])
        valid_mask = jnp.ones(params.shape[0])
        start_time = time.perf_counter()
        rollout_steps = 0
        sim_steps = 0
        for i in range(task_max_steps):
            actions, policy_state = policy_act_func(task_state, params, policy_state)
            task_state, reward, done = task_step_func(task_state, actions)
            scores, valid_mask = update_score_and_mask(scores, reward, valid_mask, done)
            rollout_steps += 1
            sim_steps = sim_steps + valid_mask
            if all_done(valid_mask):
                break
        time_cost = time.perf_counter() - start_time
        self._logger.debug(
            "{} steps/s, mean.steps={}".format(
                int(rollout_steps * task_state.obs.shape[0] / time_cost), sim_steps.sum() / task_state.obs.shape[0]
            )
        )

        return report_score(scores, n_repeats), task_state

    def _scan_loop_eval(self, params: jnp.ndarray, test: bool) -> Tuple[jnp.ndarray, TaskState, Tuple[Dict, Dict]]:
        """Rollout using jax.lax.scan."""
        policy_reset_func = self._policy_reset_fn
        if test:
            n_repeats = self._test_n_repeats
            task_reset_func = self._valid_reset_fn
            rollout_func = self._valid_rollout_fn
            params = monkey_duplicate_params(params, self._n_evaluations, False, batch=True)
        else:
            n_repeats = self._n_repeats
            task_reset_func = self._train_reset_fn
            rollout_func = self._train_rollout_fn

        # Suppose pop_size=2 and n_repeats=3.
        # For multi-agents training, params become
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        # For non-ma training, params become
        #   a1, a2, ..., an  (individual 1 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   a1, a2, ..., an  (individual 1 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   b1, b2, ..., bn  (individual 2 params)
        #   b1, b2, ..., bn  (individual 2 params)
        params = monkey_duplicate_params(params, n_repeats, self._ma_training)

        # Do the rollouts.
        if self._use_backprop and not test:
            diff_params, static_params = params
            opt_state = self._optimizer.init(diff_params)

            def model_loss_for_grad(model_params, task_state, policy_state):
                full_params = (model_params, static_params)
                scores, all_obs, masks, final_states = rollout_func(
                    task_state, policy_state, full_params, self.obs_params
                )
                # The primary objective is to maximize scores, so the loss is the negative mean of scores.
                loss = -jnp.mean(scores)

                # Add L2 penalty to the loss.
                if self._l2_penalty > 0.0:
                    # For each parameter tensor, sum the squares along all axes except the first (population) axis.
                    per_param_l2 = jax.tree_util.tree_map(
                        lambda p: jnp.sum(jnp.square(p), axis=range(1, p.ndim)), model_params
                    )

                    # Sum the L2 norms across all parameters (weights and biases) to get the total L2 norm.
                    # The result is a vector of shape (pop_size,).
                    l2_loss_per_network = sum(jax.tree_util.tree_leaves(per_param_l2))

                    # The final L2 penalty is the mean of the per-network L2 losses.
                    loss += self._l2_penalty * jnp.mean(l2_loss_per_network)

                return loss

            @jax.jit
            def update(current_params, current_opt_state, task_state, policy_state):
                loss_value, grads = jax.value_and_grad(model_loss_for_grad)(current_params, task_state, policy_state)
                updates, new_opt_state = self._optimizer.update(grads, current_opt_state)
                new_params = optax.apply_updates(current_params, updates)
                return new_params, new_opt_state, loss_value

            for epoch in range(self._backprop_steps + 1):
                self._key, reset_keys = get_task_reset_keys(
                    self._key, test, self._pop_size, self._n_evaluations, n_repeats, self._ma_training
                )

                # Reset the tasks and the policy.
                task_state = task_reset_func(reset_keys)
                policy_state = policy_reset_func(task_state)
                if self._num_device > 1:
                    params = split_params_for_pmap(params)
                    task_state = split_states_for_pmap(task_state)
                    policy_state = split_states_for_pmap(policy_state)

                diff_params, opt_state, loss = update(diff_params, opt_state, task_state, policy_state)

                # if epoch % 10 == 0:
                #     self._logger.info(f"Epoch {epoch}/{self._backprop_steps}, Loss: {loss:.4f}")

            params = (diff_params, static_params)

        self._key, reset_keys = get_task_reset_keys(
            self._key, test, self._pop_size, self._n_evaluations, n_repeats, self._ma_training
        )

        # Reset the tasks and the policy.
        task_state = task_reset_func(reset_keys)
        policy_state = policy_reset_func(task_state)
        if self._num_device > 1:
            params = split_params_for_pmap(params)
            task_state = split_states_for_pmap(task_state)
            policy_state = split_states_for_pmap(policy_state)

        scores, all_obs, masks, final_states = rollout_func(task_state, policy_state, params, self.obs_params)

        if self._num_device > 1:
            all_obs = reshape_data_from_pmap(all_obs)
            masks = reshape_data_from_pmap(masks)
            final_states = merge_state_from_pmap(final_states)

        if not test and not self.obs_normalizer.is_dummy:
            self.obs_params = self.obs_normalizer.update_normalization_params(
                obs_buffer=all_obs, obs_mask=masks, obs_params=self.obs_params
            )

        if self._ma_training:
            if not test:
                # In training, each agent has different parameters.
                scores = jnp.mean(scores.ravel().reshape((n_repeats, -1)), axis=0)
            else:
                # In tests, they share the same parameters.
                scores = jnp.mean(scores.ravel().reshape((n_repeats, -1)), axis=1)
        else:
            scores = jnp.mean(scores.ravel().reshape((-1, n_repeats)), axis=-1)

        # Note: QD methods do not support ma_training for now.
        if not self._ma_training:
            final_states = tree_map(lambda x: x.reshape((scores.shape[0], n_repeats, *x.shape[1:])), final_states)

        return scores, self._bd_summarize_fn(final_states), params
