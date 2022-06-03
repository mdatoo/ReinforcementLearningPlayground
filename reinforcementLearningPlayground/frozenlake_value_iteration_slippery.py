import gym

from reinforcementLearningPlayground.actions.action_selectors import MaxSelector, RandomSelector
from reinforcementLearningPlayground.agents import ValueIteration
from reinforcementLearningPlayground.episodes.episode_generators import ValueGenerator

BATCH_SIZE = 64


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)

    action_selector = RandomSelector(MaxSelector(), 0.01)
    episode_generator = ValueGenerator(action_selector, env)

    agent = ValueIteration(BATCH_SIZE, episode_generator)
    agent.train(0.7)
