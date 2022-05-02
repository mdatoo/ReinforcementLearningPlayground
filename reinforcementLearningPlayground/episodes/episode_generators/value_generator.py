from collections import defaultdict

import numpy as np
from gym import Env
from gym.spaces import Discrete

from reinforcementLearningPlayground.actions.action_selectors import ActionSelector
from reinforcementLearningPlayground.episodes import Step
from reinforcementLearningPlayground.episodes.episode_generators import EpisodeGenerator


class ValueGenerator(EpisodeGenerator):
    def __init__(self, action_selector: ActionSelector[np.ndarray, int], env: Env):
        super().__init__(action_selector, env)
        assert type(env.action_space) == Discrete, "ValueGenerator only works on environments with" \
                                                   " discrete action spaces"

        self.expected_rewards = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def generate_action_probs(self, state: Step.state) -> np.ndarray:
        return self.expected_rewards[state]

    def step(self) -> None:
        self.action_selector.step()
