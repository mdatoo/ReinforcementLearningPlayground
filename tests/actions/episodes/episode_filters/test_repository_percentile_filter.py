import unittest
from unittest.mock import MagicMock

from reinforcementLearningPlayground.episodes.episode_filters import RepositoryPercentileFilter


class TestRepositoryPercentileFilter(unittest.TestCase):
    def setUp(self) -> None:
        episode_repository = MagicMock()
        episode_repository.boundary = MagicMock(return_value=0.5)
        self.repository_percentile_filter = RepositoryPercentileFilter(episode_repository, 50)

    def test_filter(self) -> None:
        rewards = [0.2, 0.8, 0.4, 0.6, 1.0, 0.0]
        episodes = [MagicMock(reward=reward) for reward in rewards]

        assert self.repository_percentile_filter.filter(episodes) == [episodes[1], episodes[3], episodes[4]]


if __name__ == "__main__":
    unittest.main()
