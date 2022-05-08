from __future__ import annotations
from collections import namedtuple
from statistics import mean
from typing import List

Step = namedtuple("Step", field_names=["state", "action", "new_state", "reward", "is_done"])


class Episode:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    @property
    def reward(self):
        return sum([step.reward for step in self.steps])

    @staticmethod
    def average_reward(episodes: List[Episode]) -> float:
        if len(episodes) == 0:
            return 0.0
        return mean([episode.reward for episode in episodes])

    def __eq__(self, other):
        return type(other) == Episode and self.steps == other.steps
