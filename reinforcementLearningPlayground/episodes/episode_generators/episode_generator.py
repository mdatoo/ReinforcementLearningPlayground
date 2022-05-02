from typing import List, Generic, TypeVar

from gym import Env

from .. import Episode, Step
from ..episode_filters import EpisodeFilter
from ...actions.action_selectors import ActionSelector

A = TypeVar('A')
B = TypeVar('B')


class EpisodeGenerator:
    def __init__(self, action_selector: ActionSelector[A, B], env: Env):
        self.action_selector = action_selector
        self.env = env

    def generate(self) -> Episode:
        is_done = False
        steps = []
        state = self.env.reset()

        while not is_done:
            action_probs = self.generate_action_probs(state)
            action = self.action_selector.select(action_probs)
            new_state, reward, is_done, _ = self.env.step(action)
            steps.append(Step(state, action, new_state, reward))

            state = new_state

        return Episode(steps)

    def generate_action_probs(self, state: Step.state) -> A:
        raise NotImplementedError

    def generate_n(self, n: int) -> List[Episode]:
        return [self.generate() for _ in range(n)]

    def generate_n_valid(self, n: int, episode_filters: List[EpisodeFilter]):
        episodes = []
        valid_episodes = []

        while len(valid_episodes) < n:
            episodes.append(self.generate())
            valid_episodes = EpisodeFilter.apply(episode_filters, episodes)

        return valid_episodes

    def step(self) -> None:
        pass
