import unittest
from unittest.mock import MagicMock, call

from reinforcementLearningPlayground.episodes.episode_repositories import EpisodeRepository


class TestEpisodeRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.episode_repository = EpisodeRepository([])

    def test_extend(self) -> None:
        self.episode_repository.add = MagicMock()
        episodes = [MagicMock() for _ in range(10)]

        self.episode_repository.extend(episodes)

        calls = [call(episode) for episode in episodes]
        self.episode_repository.add.assert_has_calls(calls)

    def test_get_n(self) -> None:
        self.episode_repository.get = MagicMock()

        self.episode_repository.get_n(10)
        assert self.episode_repository.get.call_count == 10

    def test_boundary(self) -> None:
        episodes = [MagicMock(reward=reward) for reward in [1, 2, 3]]
        self.episode_repository.episodes = episodes

        assert self.episode_repository.boundary(0.6) == 1.012


if __name__ == "__main__":
    unittest.main()
