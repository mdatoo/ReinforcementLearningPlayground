from typing import List

import numpy as np

from .episode_filter import EpisodeFilter
from .. import Episode


class PercentileFilter(EpisodeFilter):
    def __init__(self, percentile: int):
        assert 0 <= percentile <= 100, f"Invalid percentile {percentile}, must be between 0 and 100 inclusive"

        self.percentile = percentile

    def filter(self, episodes: List[Episode]) -> List[Episode]:
        reward_boundary = np.percentile([episode.reward for episode in episodes], self.percentile)
        return [episode for episode in episodes if episode.reward >= reward_boundary]
