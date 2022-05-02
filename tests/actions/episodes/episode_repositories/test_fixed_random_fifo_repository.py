import unittest
from copy import copy

from reinforcementLearningPlayground.episodes import Episode
from reinforcementLearningPlayground.episodes.episode_repositories import FixedRandomFifoRepository


class TestFixedRandomFifoRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = [Episode([]) for _ in range(10)]
        self.fixed_random_fifo_repository = FixedRandomFifoRepository(copy(self.episodes))

    def test_add(self) -> None:
        episode = Episode([])
        self.fixed_random_fifo_repository.add(episode)
        assert self.fixed_random_fifo_repository.episodes == self.episodes[1:] + [episode]

    def test_get(self) -> None:
        assert self.fixed_random_fifo_repository.get() in self.episodes


if __name__ == "__main__":
    unittest.main()
