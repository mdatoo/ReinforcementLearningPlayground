import unittest
from unittest.mock import MagicMock

from reinforcementLearningPlayground.episodes.episode_filters import ThresholdFilter


class TestThresholdFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.threshold_filter = ThresholdFilter(0.6)

    def test_filter(self) -> None:
        rewards = [0.2, 0.8, 0.4, 0.6, 1.0, 0.0]
        episodes = [MagicMock(reward=reward) for reward in rewards]

        assert self.threshold_filter.filter(episodes) == [episodes[1], episodes[3], episodes[4]]

    def test_hard_filter(self) -> None:
        self.threshold_filter.hard_threshold = True

        rewards = [0.2, 0.8, 0.4, 0.6, 1.0, 0.0]
        episodes = [MagicMock(reward=reward) for reward in rewards]

        assert self.threshold_filter.filter(episodes) == [episodes[1], episodes[4]]


if __name__ == "__main__":
    unittest.main()
