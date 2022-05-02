import unittest
from unittest.mock import MagicMock

from gym.spaces import Discrete

from reinforcementLearningPlayground.episodes.episode_generators import ValueGenerator


class TestValueGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.action_selector = MagicMock()
        env = MagicMock(action_space=Discrete(2))
        self.value_generator = ValueGenerator(self.action_selector, env)

    def test_generate_action_probs(self) -> None:
        action_probs, state = object(), object()
        self.value_generator.expected_rewards[state] = action_probs

        assert self.value_generator.generate_action_probs(state) == action_probs

    def test_step(self) -> None:
        self.value_generator.step()
        self.action_selector.step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
