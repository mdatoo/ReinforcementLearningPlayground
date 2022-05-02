import gym
import numpy as np


class OneHotDecodeAction(gym.ActionWrapper):
    def action(self, action: np.ndarray) -> int:
        return np.argmax(action)

    def reverse_action(self, action: np.ndarray) -> int:
        return self.action(action)
