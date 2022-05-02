import gym

from reinforcementLearningPlayground.actions.action_selectors import WeightedSelector
from reinforcementLearningPlayground.agents import ValueIteration
from reinforcementLearningPlayground.episodes.episode_generators import ValueGenerator
from reinforcementLearningPlayground.wrappers import DiscountReward

BATCH_SIZE = 64


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = DiscountReward(env, 0.99)

    action_selector = WeightedSelector()
    episode_generator = ValueGenerator(action_selector, env)

    agent = ValueIteration(BATCH_SIZE, episode_generator)
    agent.train(0.8)
