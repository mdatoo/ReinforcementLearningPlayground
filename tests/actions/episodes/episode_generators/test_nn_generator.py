import unittest
from unittest.mock import MagicMock

import numpy as np

from reinforcementLearningPlayground.episodes.episode_generators import NnGenerator


class TestNnGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.action_selector = MagicMock()
        env = MagicMock()
        self.model = MagicMock()
        self.nn_generator = NnGenerator(self.action_selector, env, self.model)

    def test_generate_action_probs(self) -> None:
        action_probs, state = object(), object()
        self.model.return_value = [action_probs]

        assert self.nn_generator.generate_action_probs(state) == action_probs
        self.model.assert_called_once_with(np.array([state]), training=False)

    def test_step(self) -> None:
        self.nn_generator.step()
        self.action_selector.step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
