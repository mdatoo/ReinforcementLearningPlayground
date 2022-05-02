from typing import List

from .episode_filter import EpisodeFilter
from .. import Episode


class ThresholdFilter(EpisodeFilter):
    def __init__(self, threshold: float, hard_threshold: bool = False):
        self.hard_threshold = hard_threshold
        self.threshold = threshold

    def filter(self, episodes: List[Episode]) -> List[Episode]:
        if self.hard_threshold:
            return [episode for episode in episodes if episode.reward > self.threshold]
        return [episode for episode in episodes if episode.reward >= self.threshold]
