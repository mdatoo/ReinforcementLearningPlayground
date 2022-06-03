from typing import List

import numpy as np

from .. import Episode, Step


class EpisodeRepository:
    def __init__(self, initial_episodes: List[Episode]):
        self.episodes = initial_episodes

    def add(self, episode: Episode) -> None:
        raise NotImplementedError

    def extend(self, episodes: List[Episode]) -> None:
        [self.add(episode) for episode in episodes]

    def get(self) -> Episode:
        raise NotImplementedError

    def get_step(self) -> Step:
        steps = self.get().steps
        return steps[np.random.choice(len(steps))]

    def get_n(self, n: int) -> List[Episode]:
        return [self.get() for _ in range(n)]

    def get_n_steps(self, n: int) -> List[Step]:
        return [self.get_step() for _ in range(n)]

    def boundary(self, percentile: float) -> float:
        return np.percentile([episode.reward for episode in self.episodes], percentile)
