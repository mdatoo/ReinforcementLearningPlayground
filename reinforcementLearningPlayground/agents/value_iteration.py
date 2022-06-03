from collections import defaultdict
from typing import List

import numpy as np

from reinforcementLearningPlayground.agents import Agent
from ..episodes import Episode, Step

from ..episodes.episode_generators import ValueGenerator
from ..helpers import mean


class ValueIteration(Agent):
    def __init__(
            self,
            batch_size: int,
            episode_generator: ValueGenerator,
            alpha: float = 0.2,
            gamma: int = 0.9
    ):
        super().__init__(batch_size, episode_generator)

        self.alpha = alpha
        self.gamma = gamma
        self._expected_rewards = episode_generator.expected_rewards
        self._states = set()
        self._transitions = defaultdict(lambda: [])

    def train_step(self, episodes: List[Episode]) -> None:
        for episode in episodes:
            for step in episode.steps:
                self._states.add(step.state)
                self._transitions[(step.state, step.action)].append((step.new_state, step.reward))

        self._regenerate_expected_rewards()

    def _regenerate_expected_rewards(self):
        expected_rewards = {state: self._generate_expected_rewards(state) for state in self._states}
        self._expected_rewards.update(expected_rewards)

    def _generate_expected_rewards(self, state: Step.state):
        return np.array([self.alpha * self._generate_expected_reward(state, action) +
                         (1 - self.alpha) * self._expected_rewards[state][action]
                         for action in range(self.episode_generator.env.action_space.n)])

    def _generate_expected_reward(self, state: Step.state, action: Step.action) -> float:
        return mean([reward + self.gamma * self._expected_rewards[new_state]
                     for new_state, reward in self._transitions[(state, action)]])
