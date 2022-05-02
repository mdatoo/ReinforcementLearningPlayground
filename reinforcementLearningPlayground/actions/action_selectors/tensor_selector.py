import numpy as np
import tensorflow as tf

from .action_selector import ActionSelector


class TensorSelector(ActionSelector[tf.Tensor, np.ndarray]):
    def __init__(self, wrapped_selector: ActionSelector[np.ndarray, int]):
        self.wrapped_selector = wrapped_selector

    def select(self, logits: tf.Tensor) -> np.ndarray:
        ratios = logits.numpy().astype(np.float32)
        action = np.zeros_like(ratios)
        action[self.wrapped_selector.select(ratios)] = 1

        return action

    def step(self) -> None:
        self.wrapped_selector.step()
