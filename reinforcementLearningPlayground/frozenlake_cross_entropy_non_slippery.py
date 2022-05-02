import gym
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from reinforcementLearningPlayground.actions.action_selectors import WeightedSelector, TensorSelector
from reinforcementLearningPlayground.wrappers import DiscountReward, OneHotDecodeAction, OneHotEncodeObservation
from reinforcementLearningPlayground.agents import CrossEntropy
from reinforcementLearningPlayground.episodes.episode_filters import RepositoryPercentileFilter, ThresholdFilter
from reinforcementLearningPlayground.episodes.episode_generators import NnGenerator
from reinforcementLearningPlayground.episodes.episode_repositories import FixedRandomFifoRepository
from reinforcementLearningPlayground.models import Linear

ACTIONS = 4
BATCH_SIZE = 64
REPOSITORY_SIZE = BATCH_SIZE * 8
ELITE_PERCENTILE = 50


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = DiscountReward(env, 0.99)
    env = OneHotDecodeAction(env)
    env = OneHotEncodeObservation(env)

    model = Linear.build(128, ACTIONS)
    optimiser = SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])

    action_selector = TensorSelector(WeightedSelector())
    episode_generator = NnGenerator(action_selector, env, model)

    initial_episodes = episode_generator.generate_n_valid(REPOSITORY_SIZE, [ThresholdFilter(0, hard_threshold=True)])
    episode_repository = FixedRandomFifoRepository(initial_episodes)
    episode_filters = [RepositoryPercentileFilter(episode_repository, ELITE_PERCENTILE)]

    agent = CrossEntropy(BATCH_SIZE, episode_filters, episode_generator, episode_repository, model)
    agent.train(0.5)
