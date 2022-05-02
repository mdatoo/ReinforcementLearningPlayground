from typing import List

from .episode_filter import EpisodeFilter
from .. import Episode
from ..episode_repositories import EpisodeRepository


class RepositoryPercentileFilter(EpisodeFilter):
    def __init__(self, episode_repository: EpisodeRepository, percentile: int):
        assert 0 <= percentile <= 100, f"Invalid percentile {percentile}, must be between 0 and 100 inclusive"

        self.episode_repository = episode_repository
        self.percentile = percentile

    def filter(self, episodes: List[Episode]) -> List[Episode]:
        reward_boundary = self.episode_repository.boundary(self.percentile)
        return [episode for episode in episodes if episode.reward >= reward_boundary]
