import numpy as np
from gym import Env
from tensorflow.python.keras.models import Model

import tensorflow as tf

from .episode_generator import EpisodeGenerator
from .. import Step
from ...actions.action_selectors import TensorSelector


class NnGenerator(EpisodeGenerator):
    def __init__(self, tensor_selector: TensorSelector, env: Env, model: Model):
        super().__init__(tensor_selector, env)
        self.model = model

    def generate_action_probs(self, state: Step.state) -> tf.Tensor:
        return self.model(np.expand_dims(state, axis=0), training=False)[0]

    def step(self) -> None:
        self.action_selector.step()
