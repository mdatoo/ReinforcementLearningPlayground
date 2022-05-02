import unittest
from unittest.mock import MagicMock

from reinforcementLearningPlayground.episodes import Episode, Step


class TestEpisode(unittest.TestCase):
    def test_reward(self) -> None:
        rewards = [5, 4, 0, 7]
        steps = [Step(None, None, None, reward) for reward in rewards]
        episode = Episode(steps)

        assert episode.reward == sum(rewards)

    def test_average_reward(self) -> None:
        rewards = [5, 4, 0, 7]
        episodes = [MagicMock(reward=reward) for reward in rewards]

        assert Episode.average_reward(episodes) == sum(rewards) / len(rewards)

    def test_average_reward_empty(self) -> None:
        assert Episode.average_reward([]) == 0


if __name__ == "__main__":
    unittest.main()
