import gym
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from reinforcementLearningPlayground.actions.action_selectors import TensorSelector, RandomSelector, \
    MaxSelector
from reinforcementLearningPlayground.agents import DeepQ
from reinforcementLearningPlayground.episodes.episode_generators import NnGenerator
from reinforcementLearningPlayground.episodes.episode_repositories import FixedRandomFifoRepository
from reinforcementLearningPlayground.models import Linear
from reinforcementLearningPlayground.wrappers import OneHotDecodeAction, OneHotEncodeObservation

ACTIONS = 4
BATCH_SIZE = 128
REPOSITORY_SIZE = 128
ELITE_PERCENTILE = 70


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env = OneHotDecodeAction(env)
    env = OneHotEncodeObservation(env)

    model = Linear.build(128, ACTIONS)
    loss = CategoricalCrossentropy()
    optimiser = SGD(learning_rate=0.01)

    action_selector = TensorSelector(RandomSelector(MaxSelector(), 0.001))
    episode_filters = []
    episode_generator = NnGenerator(action_selector, env, model)

    initial_episodes = episode_generator.generate_n_valid(REPOSITORY_SIZE, episode_filters)
    episode_repository = FixedRandomFifoRepository(initial_episodes)

    agent = DeepQ(BATCH_SIZE, episode_filters, episode_generator, episode_repository, loss, optimiser, sync_frames=10)
    agent.train(0.5)
