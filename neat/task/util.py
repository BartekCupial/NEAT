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


def render_sailency_map(obs, labels, policy_action_fn):
    plt.figure(figsize=(8, 8))

    # Create a grid of points to cover the observation space
    x_min, x_max = obs[:, 0].min() - 0.5, obs[:, 0].max() + 0.5
    y_min, y_max = obs[:, 1].min() - 0.5, obs[:, 1].max() + 0.5
    resolution = 150
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Query the policy for each point on the grid to find the decision boundary
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]
    # The provided policy_action_fn is used here to get predictions for the grid
    grid_predictions = policy_action_fn(grid_points)

    # Calculate logit difference to determine color intensity
    logits_class_0 = grid_predictions[:, 0]
    logits_class_1 = grid_predictions[:, 1]
    diff_logits = np.abs(logits_class_1 - logits_class_0)

    # Normalize the difference to a [0, 1] range for blending
    min_diff, max_diff = diff_logits.min(), diff_logits.max()
    norm_diff = (diff_logits - min_diff) / (max_diff - min_diff + 1e-8)

    # Create an RGB color grid for plotting
    orange = np.array([232 / 255, 175 / 255, 108 / 255])  # RGB for orange
    blue = np.array([86 / 255, 146 / 255, 185 / 255])  # RGB for blue
    white = np.array([1.0, 1.0, 1.0])

    is_class_1_dominant = logits_class_1 > logits_class_0
    base_colors = np.where(is_class_1_dominant[:, np.newaxis], blue, orange)

    # Blend base color with white based on confidence (norm_diff)
    # norm_diff is the weight of the base color. (1 - norm_diff) is the weight of white.
    blend_factor = norm_diff[:, np.newaxis]
    color_grid = blend_factor * base_colors + (1 - blend_factor) * white
    color_grid_reshaped = color_grid.reshape((resolution, resolution, 3))

    # Plot the color grid and the ground truth points
    plt.imshow(color_grid_reshaped, extent=(x_min, x_max, y_min, y_max), origin="lower", alpha=0.8)

    plt.plot(
        obs[labels == 0, 0], obs[labels == 0, 1], "o", label="Class 0", ms=6, color="#E8AF6C", markeredgecolor="white"
    )
    plt.plot(
        obs[labels == 1, 0], obs[labels == 1, 1], "o", label="Class 1", ms=6, color="#5692B9", markeredgecolor="white"
    )

    # Finalize the plot and convert to a PIL Image
    plt.title("Two-Color Intensity of Model Output vs. Ground Truth")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    return Image.open(buf)
