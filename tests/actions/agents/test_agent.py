import unittest
from unittest.mock import MagicMock

from reinforcementLearningPlayground.agents import Agent


class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.episode_generator = MagicMock()

        self.batch_size = 64
        self.agent = Agent(self.batch_size, self.episode_generator)
        self.agent.train_step = MagicMock()

    def test_train(self) -> None:
        generate_n_return = [MagicMock(reward=0.2)]
        self.episode_generator.generate_n = MagicMock(return_value=generate_n_return)

        self.agent.train(0.2)

        self.episode_generator.generate_n.assert_called_once_with(self.batch_size)
        self.agent.train_step.assert_called_once_with(generate_n_return)
        self.episode_generator.step.assert_called_once()

    def test_run(self) -> None:
        generate_return = object()
        self.episode_generator.generate = MagicMock(return_value=generate_return)

        assert self.agent.run() == generate_return


if __name__ == "__main__":
    unittest.main()
