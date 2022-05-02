import unittest

import numpy as np

from reinforcementLearningPlayground.actions.action_selectors import WeightedSelector


class TestWeightedSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.weighted_selector = WeightedSelector()

    def test_select(self) -> None:
        ratios = np.array([0, 0.5, 0.2, 0.1])

        assert self.weighted_selector.select(ratios) < len(ratios)

    def test_select_all_0_but_one(self) -> None:
        ratios = np.array([0, 0, 1, 0])

        assert self.weighted_selector.select(ratios) == 2


if __name__ == "__main__":
    unittest.main()
