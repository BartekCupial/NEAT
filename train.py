"""Train an agent for XOR problem.

Example command to run this script: `python train_xor.py --gpu-id=0`
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

import hydra
from evojax import util
from evojax.task.slimevolley import SlimeVolley
from omegaconf import DictConfig

from neat.algo.genome import ActivationFunction
from neat.algo.neat import NEAT
from neat.policy import NEATPolicy
from neat.task import XOR, Circle, Spiral
from neat.trainer import NEATTrainer


def get_task(config: DictConfig, test: bool = False):
    """Get the task based on the configuration."""
    if config.task.name == "xor":
        return XOR(batch_size=config.task.batch_size, dataset_size=config.task.dataset_size, test=test)
    elif config.task.name == "circle":
        return Circle(batch_size=config.task.batch_size, dataset_size=config.task.dataset_size, test=test)
    elif config.task.name == "spiral":
        return Spiral(batch_size=config.task.batch_size, dataset_size=config.task.dataset_size, test=test)
    elif config.task.name == "slimevolley":
        return SlimeVolley(
            batch_size=config.task.batch_size,
            dataset_size=config.task.dataset_size,
            max_steps=config.task.max_steps,
            test=test,
        )
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

    logger = util.create_logger(name=config.task.name, log_dir=output_dir, debug=config.eval.debug)
    logger.info(f"EvoJAX {config.task} Demo")
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
        activation_function=ActivationFunction.TANH,
        last_activation_function=ActivationFunction.IDENTITY,
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
        n_repeats=1,
        n_evaluations=1,
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


if __name__ == "__main__":
    main()
