import numpy as np

from .episode_repository import EpisodeRepository
from .. import Episode


class FixedRandomFifoRepository(EpisodeRepository):
    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        self.episodes.pop(0)

    def get(self) -> Episode:
        return np.random.choice(self.episodes)
