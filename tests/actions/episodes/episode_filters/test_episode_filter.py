import unittest
from unittest.mock import MagicMock

from reinforcementLearningPlayground.episodes.episode_filters import EpisodeFilter


class TestEpisodeFilter(unittest.TestCase):
    def test_apply(self) -> None:
        episodes = [MagicMock() for i in range(5)]
        episode_filters = [MagicMock() for i in range(3)]
        for episode_filter in episode_filters:
            episode_filter.filter = lambda l: l[1:]

        assert EpisodeFilter.apply(episode_filters, episodes) == episodes[3:]


if __name__ == "__main__":
    unittest.main()
