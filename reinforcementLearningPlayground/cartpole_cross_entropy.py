import gym
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from reinforcementLearningPlayground.actions.action_selectors import WeightedSelector, TensorSelector
from reinforcementLearningPlayground.wrappers import OneHotDecodeAction
from reinforcementLearningPlayground.agents import CrossEntropy
from reinforcementLearningPlayground.episodes.episode_filters import PercentileFilter, ThresholdFilter
from reinforcementLearningPlayground.episodes.episode_generators import NnGenerator
from reinforcementLearningPlayground.episodes.episode_repositories import FixedRandomFifoRepository
from reinforcementLearningPlayground.models import Linear

ACTIONS = 2
BATCH_SIZE = 64
REPOSITORY_SIZE = BATCH_SIZE * 2
ELITE_PERCENTILE = 70


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env = OneHotDecodeAction(env)

    model = Linear.build(128, ACTIONS)
    optimiser = SGD(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])

    action_selector = TensorSelector(WeightedSelector())
    episode_filters = [PercentileFilter(ELITE_PERCENTILE), ThresholdFilter(0, hard_threshold=True)]
    episode_generator = NnGenerator(action_selector, env, model)

    initial_episodes = episode_generator.generate_n_valid(REPOSITORY_SIZE, episode_filters)
    episode_repository = FixedRandomFifoRepository(initial_episodes)

    agent = CrossEntropy(BATCH_SIZE, episode_filters, episode_generator, episode_repository, model)
    agent.train(env.spec.reward_threshold)
