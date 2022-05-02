from __future__ import annotations

from typing import List

from .. import Episode


class EpisodeFilter:
    def filter(self, episodes: List[Episode]) -> List[Episode]:
        raise NotImplementedError

    def step(self) -> None:
        pass

    @staticmethod
    def apply(episode_filters: List[EpisodeFilter], episodes: List[Episode]) -> List[Episode]:
        valid_episodes = episodes
        for episode_filter in episode_filters:
            valid_episodes = episode_filter.filter(valid_episodes)

        return valid_episodes
