import math

import numpy as np

from .action_selector import ActionSelector


class RandomSelector(ActionSelector[np.ndarray, int]):
    def __init__(
            self,
            wrapped_selector: ActionSelector[np.ndarray, int],
            step: float,
            start: float = 1,
            end: float = 0
    ):
        assert 0 <= start <= 1, f"Invalid start epsilon {start}, must be between 0 and 1 inclusive"
        assert 0 <= end <= 1, f"Invalid start epsilon {end}, must be between 0 and 1 inclusive"
        assert start >= end, f"Invalid end epsilon {end}, must be less than or equal to start epsilon {start}"

        self.wrapped_selector = wrapped_selector
        self.epsilon = start
        self.end_epsilon = end
        self.step_epsilon = abs(step)

    def select(self, values: np.ndarray) -> int:
        assert values.ndim == 1, "Random action selector can only pick from one dimensional arrays"

        if np.random.random() < self.epsilon:
            return np.random.randint(values.size)
        else:
            return self.wrapped_selector.select(values)

    def step(self) -> None:
        self.wrapped_selector.step()

        if self.epsilon > self.end_epsilon:
            self.epsilon -= self.step_epsilon
            self.epsilon = round(self.epsilon, math.ceil(abs(math.log10(self.step_epsilon))))  # remove rounding error

        print(f"Epsilon: {self.epsilon}")
