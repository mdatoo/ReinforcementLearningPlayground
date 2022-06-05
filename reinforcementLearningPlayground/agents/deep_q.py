from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.python.keras.models import clone_model
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2 as Optimizer
from tensorflow.python.keras.losses import Loss

from .agent import Agent
from ..episodes import Episode
from ..episodes.episode_filters import EpisodeFilter
from ..episodes.episode_generators import NnGenerator
from ..episodes.episode_repositories import EpisodeRepository


class DeepQ(Agent):
    def __init__(
            self,
            batch_size: int,
            episode_filters: List[EpisodeFilter],
            episode_generator: NnGenerator,
            episode_repository: EpisodeRepository,
            loss: Loss,
            optimiser: Optimizer,
            gamma: int = 0.9,
            sync_frames: int = 1000
    ):
        super().__init__(batch_size, episode_generator)

        self.episode_filters = episode_filters
        self.episode_repository = episode_repository

        self.model = episode_generator.model
        self.tgt_model = clone_model(self.model)
        self.loss = loss
        self.optimiser = optimiser

        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_frames = sync_frames
        self.step = 0

    def train_step(self, episodes: List[Episode]) -> None:
        if self.step % self.sync_frames == 0:
            self.tgt_model.set_weights(self.model.get_weights())
        self.step += 1

        self._update_repository(episodes)
        self._train_model()

    def _update_repository(self, episodes: List[Episode]) -> None:
        valid_episodes = EpisodeFilter.apply(self.episode_filters, episodes)
        self.episode_repository.extend(valid_episodes)

    def _train_model(self) -> None:
        steps = self.episode_repository.get_n_steps(self.batch_size)
        states = np.asarray([step.state for step in steps])
        actions = np.asarray([step.action for step in steps], dtype=int)
        new_states = np.asarray([step.new_state for step in steps])
        rewards = np.asarray([step.reward for step in steps])
        not_dones = np.asarray([0 if step.is_done else 1 for step in steps])

        with GradientTape() as tape:
            logits = self.model(states, training=True)
            predicted_q = tf.gather(logits, tf.argmax(actions, axis=1), batch_dims=1)
            actual_q = rewards + self.gamma * not_dones * tf.reduce_max(self.tgt_model(new_states, training=False),
                                                                        axis=1)
            loss = self.loss(actual_q, predicted_q)
            print(f"Loss: {loss}")

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimiser.apply_gradients(zip(grads, self.model.trainable_weights))
