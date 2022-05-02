import unittest
from unittest.mock import MagicMock

import numpy as np

from reinforcementLearningPlayground.actions.action_selectors import RandomSelector


class TestMaxSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapped_return = object()
        self.wrapped_selector = MagicMock()
        self.wrapped_selector.select = MagicMock(return_value=self.wrapped_return)

        self.random_selector_step = 0.1
        self.random_selector = RandomSelector(self.wrapped_selector, self.random_selector_step)

    def test_select_full_random(self) -> None:
        self.random_selector.epsilon = 1
        ratios = np.array([0, 0.5, 0.2, 0.1])

        selected = self.random_selector.select(ratios)
        assert selected < len(ratios)
        assert selected != self.wrapped_return

    def test_select_full_wrapped(self) -> None:
        self.random_selector.epsilon = 0
        ratios = np.array([0, 0.5, 0.2, 0.1])

        assert self.random_selector.select(ratios) == self.wrapped_return

    def test_step(self) -> None:
        self.random_selector.epsilon = 1
        self.random_selector.step()

        assert self.random_selector.epsilon == 1 - self.random_selector_step
        self.wrapped_selector.step.assert_called()


if __name__ == "__main__":
    unittest.main()
