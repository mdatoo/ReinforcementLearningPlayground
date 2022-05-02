import unittest
from unittest.mock import MagicMock

from reinforcementLearningPlayground.episodes import Episode, Step
from reinforcementLearningPlayground.episodes.episode_generators import EpisodeGenerator


class TestEpisodeGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.action_selector = MagicMock()
        self.env = MagicMock()
        self.episode_generator = EpisodeGenerator(self.action_selector, self.env)

    def test_generate(self) -> None:
        action_probs, state, action, new_state, reward = object(), object(), object(), object(), object()
        self.env.reset = MagicMock(return_value=state)
        self.episode_generator.generate_action_probs = MagicMock(return_value=action_probs)
        self.action_selector.select = MagicMock(return_value=action)
        self.env.step = MagicMock(return_value=(new_state, reward, True, None))

        assert self.episode_generator.generate() == Episode([Step(state, action, new_state, reward)])
        self.episode_generator.generate_action_probs.assert_called_once_with(state)
        self.action_selector.select.assert_called_once_with(action_probs)
        self.env.step.assert_called_once_with(action)

    def test_generate_n(self) -> None:
        self.episode_generator.generate = MagicMock(return_value=1)

        assert self.episode_generator.generate_n(10) == [1 for _ in range(10)]
        assert self.episode_generator.generate.call_count == 10

    def test_generate_n_valid(self) -> None:
        self.episode_generator.generate = MagicMock(return_value=1)

        episode_filter = MagicMock()
        episode_filter.filter = lambda e: e[1::2]

        assert self.episode_generator.generate_n_valid(10, [episode_filter]) == [1 for _ in range(10)]
        assert self.episode_generator.generate.call_count == 20


if __name__ == "__main__":
    unittest.main()
