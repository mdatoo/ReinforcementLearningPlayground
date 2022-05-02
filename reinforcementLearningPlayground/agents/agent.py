from typing import List

from ..episodes import Episode
from ..episodes.episode_generators import EpisodeGenerator


class Agent:
    def __init__(self, batch_size: int, episode_generator: EpisodeGenerator):
        self.batch_size = batch_size
        self.episode_generator = episode_generator

    def train(self, reward_boundary: float) -> None:
        reward = 0

        while reward < reward_boundary:
            episodes = self.episode_generator.generate_n(self.batch_size)
            self.train_step(episodes)

            self.episode_generator.step()
            reward = Episode.average_reward(episodes)
            print(f"Reward: {reward}")

    def train_step(self, episodes: List[Episode]) -> None:
        raise NotImplementedError

    def run(self) -> Episode:
        return self.episode_generator.generate()
