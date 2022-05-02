import unittest

import numpy as np

from reinforcementLearningPlayground.helpers import mean, normalize, resize_image


class TestProcessing(unittest.TestCase):
    def test_mean(self) -> None:
        assert mean([0, 1, 2.5, 0.5]) == 1

    def test_mean_empty(self) -> None:
        assert mean([]) == 0

    def test_normalize(self) -> None:
        assert (normalize(np.array([[0, 1], [3, 0]])) == np.array([[0, 0.25], [0.75, 0]])).all()

    def test_normalize_zeros(self) -> None:
        assert (normalize((np.zeros((3, 2)))) == 1/6).all()

    def test_resize_image(self) -> None:
        assert resize_image(np.ndarray((3, 2)), (2, 2)).shape == (2, 2)


if __name__ == "__main__":
    unittest.main()
