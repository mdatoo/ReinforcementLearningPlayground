import gym

from .agent import Agent
from .cross_entropy import CrossEntropy
from .value_iteration import ValueIteration
from ..environments.dribbling.v0 import BASE_REWARD as DV0_BASE_REWARD


gym.register(
    id="Dribbling-v0",
    entry_point="gymPymunk.environments.dribbling.v0:Environment",
    max_episode_steps=100,
    reward_threshold=DV0_BASE_REWARD
)
