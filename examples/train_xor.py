"""Train an agent for XOR problem.

Example command to run this script: `python train_xor.py --gpu-id=0`
"""

import argparse
import os
import shutil

from evojax import util

from neat.algo import NEAT
from neat.policy import NEATPolicy
from neat.task import XOR
from neat.trainer import NEATTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=64, help="NE population size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--dataset-size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Max training iterations.")
    parser.add_argument("--test-interval", type=int, default=1000, help="Test interval.")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval.")
    parser.add_argument("--c1", type=float, default=1.0, help="NEAT c1 parameter.")
    parser.add_argument("--c2", type=float, default=1.0, help="NEAT c2 parameter.")
    parser.add_argument("--c3", type=float, default=0.4, help="NEAT c3 parameter.")
    parser.add_argument("--compatibility-threshold", type=float, default=3.0, help="NEAT compatibility threshold.")
    parser.add_argument("--survival-threshold", type=float, default=0.2, help="NEAT survival threshold.")
    parser.add_argument("--backprop-steps", type=int, default=20, help="Number of backpropagation steps.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for backpropagation.")
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"], help="Optimizer type."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("--gpu-id", type=str, help="GPU(s) to use.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = "./log/xor"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(name="XOR", log_dir=log_dir, debug=config.debug)
    logger.info("EvoJAX XOR Demo")
    logger.info("=" * 30)

    policy = NEATPolicy()
    train_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=False)
    test_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=True)
    solver = NEAT(
        pop_size=config.pop_size,
        num_inputs=2,  # XOR has 2 inputs
        num_outputs=2,  # XOR has 2 outputs
        survival_threshold=config.survival_threshold,  # NEAT-specific parameter
        compatibility_threshold=config.compatibility_threshold,  # For speciation
        c1=config.c1,
        c2=config.c2,
        c3=config.c3,
        logger=logger,
        seed=config.seed,
    )

    # Train.
    trainer = NEATTrainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=1,
        n_evaluations=1,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == "__main__":
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_id
    main(configs)
