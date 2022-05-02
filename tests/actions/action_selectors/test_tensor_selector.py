import unittest
from unittest.mock import MagicMock

import tensorflow as tf

from reinforcementLearningPlayground.actions.action_selectors import TensorSelector


class TestTensorSelector(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapped_return = 1
        self.wrapped_selector = MagicMock()
        self.wrapped_selector.select = MagicMock(return_value=self.wrapped_return)

        self.tensor_selector = TensorSelector(self.wrapped_selector)

    def test_select(self) -> None:
        ratios = tf.constant([0, 0.5, 0.2, 0.1])

        assert (self.tensor_selector.select(ratios) == [0, 1, 0, 0]).all()

    def test_step(self) -> None:
        self.tensor_selector.step()
        self.wrapped_selector.step.assert_called()
