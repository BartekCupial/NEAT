"""Train an agent for XOR problem.

Example command to run this script: `python train_xor.py --gpu-id=0`
"""

import argparse
import os
import shutil

from evojax import util
from evojax.algo import PGPE
from evojax.policy.mlp import MLPPolicy
from evojax.trainer import Trainer

from neat.task import XOR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=64, help="NE population size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--dataset-size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--max-iter", type=int, default=5000, help="Max training iterations.")
    parser.add_argument("--test-interval", type=int, default=1000, help="Test interval.")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training.")
    parser.add_argument("--center-lr", type=float, default=0.006, help="Center learning rate.")
    parser.add_argument("--std-lr", type=float, default=0.089, help="Std learning rate.")
    parser.add_argument("--init-std", type=float, default=0.039, help="Initial std.")
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

    policy = MLPPolicy(input_dim=2, hidden_dims=[4], output_dim=2, output_act_fn="tanh", logger=logger)
    train_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=False)
    test_task = XOR(batch_size=config.batch_size, dataset_size=config.dataset_size, test=True)
    solver = PGPE(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        optimizer="adam",
        center_learning_rate=config.center_lr,
        stdev_learning_rate=config.std_lr,
        init_stdev=config.init_std,
        logger=logger,
        seed=config.seed,
    )

    # Train.
    trainer = Trainer(
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
