import numpy as np

from .action_selector import ActionSelector


class MaxSelector(ActionSelector[np.ndarray, int]):
    def select(self, ratios: np.ndarray) -> int:
        assert ratios.ndim == 1, "Maximum action selector can only pick from one dimensional arrays"
        return np.argmax(ratios)
