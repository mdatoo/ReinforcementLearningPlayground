import unittest

import numpy as np

from reinforcementLearningPlayground.actions.action_selectors import MaxSelector


class TestMaxSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.max_selector = MaxSelector()

    def test_select(self) -> None:
        ratios = np.array([0, 0.5, 0.2, 0.1])
        assert self.max_selector.select(ratios) == 1


if __name__ == "__main__":
    unittest.main()
