import numpy as np

from ...helpers import normalize
from .action_selector import ActionSelector


class WeightedSelector(ActionSelector[np.ndarray, int]):
    def select(self, ratios: np.ndarray) -> int:
        assert ratios.ndim == 1, "Weighted selector can only pick from one dimensional arrays"

        probabilities = normalize(ratios)
        return np.random.choice(ratios.size, p=probabilities)
