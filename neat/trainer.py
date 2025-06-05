import logging
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
from evojax.algo import NEAlgorithm
from evojax.obs_norm import ObsNormalizer
from evojax.policy import PolicyNetwork
from evojax.task.base import VectorizedTask
from evojax.trainer import Trainer
from evojax.util import create_logger

from neat.sim_mgr import BackpropSimManager


class NEATTrainer(Trainer):
    """A trainer for NEAT algorithms."""

    def __init__(
        self,
        policy: PolicyNetwork,
        solver: NEAlgorithm,
        train_task: VectorizedTask,
        test_task: VectorizedTask,
        max_iter: int = 1000,
        log_interval: int = 20,
        test_interval: int = 100,
        n_repeats: int = 1,
        test_n_repeats: int = 1,
        n_evaluations: int = 100,
        seed: int = 42,
        debug: bool = False,
        use_for_loop: bool = False,
        normalize_obs: bool = False,
        model_dir: str = None,
        log_dir: str = None,
        logger: logging.Logger = None,
        log_scores_fn: Optional[Callable[[int, jnp.ndarray, str], None]] = None,
        use_backprop: bool = True,
        backprop_steps=20,
        learning_rate=0.001,
        optimizer="adam",
    ):
        """Initialization.

        Args:
            policy - The policy network to use.
            solver - The ES algorithm for optimization.
            train_task - The task for training.
            test_task - The task for evaluation.
            max_iter - Maximum number of training iterations.
            log_interval - Interval for logging.
            test_interval - Interval for tests.
            n_repeats - Number of rollout repetitions.
            n_evaluations - Number of tests to conduct.
            seed - Random seed to use.
            debug - Whether to turn on the debug flag.
            use_for_loop - Use for loop for rollouts.
            normalize_obs - Whether to use an observation normalizer.
            model_dir - Directory to save/load model.
            log_dir - Directory to dump logs.
            logger - Logger.
            log_scores_fn - custom function to log the scores array. Expects input:
                `current_iter`: int, `scores`: jnp.ndarray, 'stage': str = "train" | "test"
            backprop_steps - Number of gradient descent steps to perform.
            learning_rate - Learning rate for gradient descent.
            optimizer - Optimizer type ('adam', 'sgd', 'rmsprop').
        """

        if logger is None:
            self._logger = create_logger(name="Trainer", log_dir=log_dir, debug=debug)
        else:
            self._logger = logger

        self._log_interval = log_interval
        self._test_interval = test_interval
        self._max_iter = max_iter
        self.model_dir = model_dir
        self._log_dir = log_dir

        self._log_scores_fn = log_scores_fn or (lambda x, y, z: None)

        self._obs_normalizer = ObsNormalizer(
            obs_shape=train_task.obs_shape,
            dummy=not normalize_obs,
        )

        self.solver = solver
        self.sim_mgr = BackpropSimManager(
            n_repeats=n_repeats,
            test_n_repeats=test_n_repeats,
            pop_size=solver.pop_size,
            n_evaluations=n_evaluations,
            policy_net=policy,
            train_vec_task=train_task,
            valid_vec_task=test_task,
            seed=seed,
            obs_normalizer=self._obs_normalizer,
            use_for_loop=use_for_loop,
            logger=self._logger,
            use_backprop=use_backprop,
            backprop_steps=backprop_steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )
