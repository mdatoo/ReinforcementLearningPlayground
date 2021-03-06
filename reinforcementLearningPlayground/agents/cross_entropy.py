from typing import List

import numpy as np

from .agent import Agent
from ..episodes import Episode
from ..episodes.episode_filters import EpisodeFilter
from ..episodes.episode_generators import NnGenerator
from ..episodes.episode_repositories import EpisodeRepository


class CrossEntropy(Agent):
    def __init__(
            self,
            batch_size: int,
            episode_filters: List[EpisodeFilter],
            episode_generator: NnGenerator,
            episode_repository: EpisodeRepository
    ):
        super().__init__(batch_size, episode_generator)

        self.episode_filters = episode_filters
        self.episode_repository = episode_repository
        self.model = episode_generator.model
        self.batch_size = batch_size

    def train_step(self, episodes: List[Episode]) -> None:
        self._update_repository(episodes)
        self._train_model()

    def _update_repository(self, episodes: List[Episode]) -> None:
        valid_episodes = EpisodeFilter.apply(self.episode_filters, episodes)
        self.episode_repository.extend(valid_episodes)

    def _train_model(self) -> None:
        episodes = self.episode_repository.get_n(self.batch_size)
        states = np.asarray([step.state for episode in episodes for step in episode.steps])
        actions = np.asarray([step.action for episode in episodes for step in episode.steps])

        self.model.fit(states, actions, batch_size=self.batch_size)
