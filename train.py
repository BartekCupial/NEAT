"""Train an agent for XOR problem.

Example command to run this script: `python train_xor.py --gpu-id=0`
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.tree_util
from evojax import util
from evojax.task.slimevolley import SlimeVolley
from omegaconf import DictConfig, OmegaConf

from neat.algo.genome import ActivationFunction
from neat.algo.neat import NEAT
from neat.policy import NEATPolicy
from neat.sim_mgr import monkey_duplicate_params
from neat.task import XOR, Circle, Spiral
from neat.task.util import render_saliency_map
from neat.trainer import NEATTrainer, load_model


def get_task(config: DictConfig, test: bool = False):
    """Get the task based on the configuration."""
    if config.task.name == "xor":
        return XOR(batch_size=config.task.batch_size, dataset_size=config.task.dataset_size, test=test)
    elif config.task.name == "circle":
        return Circle(batch_size=config.task.batch_size, dataset_size=config.task.dataset_size, test=test)
    elif config.task.name == "spiral":
        return Spiral(batch_size=config.task.batch_size, dataset_size=config.task.dataset_size, test=test)
    elif config.task.name == "slimevolley":
        return SlimeVolley(max_steps=config.task.max_steps, test=test)
    else:
        raise ValueError(f"Unknown task: {config.task}. Supported tasks: xor, circle, spiral.")


@hydra.main(config_path="neat/config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{config.task.name}"
    output_dir = os.path.join(config.eval.output_dir, run_name)

    # Create the directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # save the configuration to a file
    config_file = os.path.join(output_dir, "config.yaml")
    with open(config_file, "w") as f:
        OmegaConf.save(config, f)

    logger = util.create_logger(name=config.task.name, log_dir=output_dir, debug=config.eval.debug)
    logger.info(json.dumps(OmegaConf.to_container(config, resolve=True), indent=4))
    logger.info("=" * 30)

    policy = NEATPolicy()
    train_task = get_task(config, test=False)
    test_task = get_task(config, test=True)
    solver = NEAT(
        pop_size=config.neat.pop_size,
        num_inputs=train_task.obs_shape[0],
        num_outputs=train_task.act_shape[0],
        survival_threshold=config.neat.survival_threshold,
        compatibility_threshold=config.neat.compatibility_threshold,
        c1=config.neat.c1,
        c2=config.neat.c2,
        c3=config.neat.c3,
        prob_add_node=config.neat.prob_add_node,
        prob_add_connection=config.neat.prob_add_connection,
        max_stagnation=config.neat.max_stagnation,
        activation_function=config.neat.activation_function,
        last_activation_function=config.neat.last_activation_function,
        logger=logger,
        seed=config.eval.seed,
        log_dir=output_dir,
    )

    # Train.
    trainer = NEATTrainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.trainer.max_iter,
        log_interval=config.trainer.log_interval,
        test_interval=config.trainer.test_interval,
        n_repeats=config.trainer.n_repeats,
        n_evaluations=config.trainer.n_evaluations,
        seed=config.eval.seed,
        log_dir=output_dir,
        use_backprop=config.trainer.use_backprop,
        backprop_steps=config.trainer.backprop_steps,
        learning_rate=config.trainer.learning_rate,
        l2_penalty=config.trainer.l2_penalty,
        complexity_penalty=config.trainer.complexity_penalty,
        optimizer=config.trainer.optimizer,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(output_dir, "best.npz")
    tar_file = os.path.join(output_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = output_dir
    trainer.run(demo_mode=True)

    # Visualize the policy.
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)

    best_params, obs_params = load_model(model_dir=output_dir)
    best_params = monkey_duplicate_params(best_params, 1, False, True)
    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)
    screens = []
    for _ in range(test_task.max_steps):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        task_state, reward, done = step_fn(task_state, action)

        # Extract the single state from the batched task_state
        current_unbatched_state = jax.tree_util.tree_map(lambda x: x[0], task_state)
        if config.task.name == "slimevolley":
            screens.append(SlimeVolley.render(current_unbatched_state))
        else:

            def policy_action_for_viz(obs):
                dummy_policy_state = policy_reset_fn(task_state)
                actions, _ = policy.get_actions(
                    jax.tree_util.tree_map(lambda x: obs, task_state), best_params, dummy_policy_state
                )
                return actions

            screens.append(
                render_saliency_map(
                    current_unbatched_state.obs,
                    current_unbatched_state.labels,
                    policy_action_for_viz,
                    blending_aggressiveness=0.5,
                )
            )

    gif_file = os.path.join(output_dir, f"{config.task.name}.gif")
    screens[0].save(gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0)
    logger.info("GIF saved to {}.".format(gif_file))


if __name__ == "__main__":
    main()
