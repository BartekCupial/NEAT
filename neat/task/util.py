import io
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from evojax.task.base import TaskState
from flax.struct import dataclass
from jax import random
from PIL import Image


@dataclass
class State(TaskState):
    obs: jnp.ndarray
    labels: jnp.ndarray


def sample_batch(key: jnp.ndarray, data: jnp.ndarray, labels: jnp.ndarray, batch_size: int) -> Tuple:
    # Ensure key has proper dimensionality
    # if key.ndim == 0:
    #     key = jnp.array([key, key])  # Convert scalar to proper key format
    # elif key.ndim == 1 and key.shape[0] != 2:
    #     key = jax.random.PRNGKey(int(key[0]))  # Regenerate proper key

    ix = random.choice(key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0), jnp.take(labels, indices=ix, axis=0))


def loss_fn(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    per_example_losses = -jnp.sum(jax.nn.log_softmax(prediction, axis=-1) * jax.nn.one_hot(target, 2), axis=-1)
    return jnp.mean(per_example_losses)


def accuracy(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.float32:
    predicted_class = jnp.argmax(prediction, axis=1)
    return jnp.mean(predicted_class == target)


def render_obs_with_ground_truth(obs, labels, actions):
    """
    Visualizes network predictions, highlighting correct and incorrect classifications.
    - Color indicates the predicted class (Blue for 0, Red for 1).
    - Marker indicates correctness ('o' for correct, 'x' for incorrect).

    Args:
        obs (np.ndarray): The observation data points (N, 2).
        labels (np.ndarray): The ground truth labels (N,).
        actions (jnp.ndarray): The network's output logits or probabilities (N, 2).

    Returns:
        PIL.Image: An image of the resulting plot.
    """
    # Determine the predicted class from the network's actions
    predicted_class = jnp.argmax(actions, axis=1)

    # Create boolean masks to identify correct and incorrect predictions
    correct_mask = predicted_class == labels
    incorrect_mask = predicted_class != labels

    # Create masks for the four possible outcomes
    correctly_predicted_0 = correct_mask & (predicted_class == 0)
    correctly_predicted_1 = correct_mask & (predicted_class == 1)
    incorrectly_predicted_as_0 = incorrect_mask & (predicted_class == 0)
    incorrectly_predicted_as_1 = incorrect_mask & (predicted_class == 1)

    plt.figure(figsize=(8, 8))

    # Plot correctly classified points (using circles 'o')
    plt.plot(obs[correctly_predicted_0, 0], obs[correctly_predicted_0, 1], "bo", label="Correct (Class 0)", ms=6)
    plt.plot(obs[correctly_predicted_1, 0], obs[correctly_predicted_1, 1], "ro", label="Correct (Class 1)", ms=6)

    # Plot incorrectly classified points (using crosses 'x')
    plt.plot(
        obs[incorrectly_predicted_as_0, 0],
        obs[incorrectly_predicted_as_0, 1],
        "bx",
        label="Incorrect (Predicted 0)",
        ms=8,
        mew=2,
    )
    plt.plot(
        obs[incorrectly_predicted_as_1, 0],
        obs[incorrectly_predicted_as_1, 1],
        "rx",
        label="Incorrect (Predicted 1)",
        ms=8,
        mew=2,
    )

    plt.title("Classification vs. Ground Truth")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Convert the plot to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_array = np.array(Image.open(buf))

    return Image.fromarray(img_array)
