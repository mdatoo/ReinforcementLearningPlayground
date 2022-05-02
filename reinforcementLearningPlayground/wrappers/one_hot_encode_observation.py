import gym
import numpy as np


class OneHotEncodeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, (env.observation_space.n,))

    def observation(self, observation: int) -> np.ndarray:
        action = np.copy(self.observation_space.low)
        action[observation] = 1

        return action
