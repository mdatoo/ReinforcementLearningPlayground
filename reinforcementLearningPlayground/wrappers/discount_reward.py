import gym


class DiscountReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env, discount_factor: float):
        super().__init__(env)
        assert 0 <= discount_factor <= 1, f"Invalid discount factor {discount_factor}, " \
                                          f"must be between 0 and 1 inclusive"

        self.discount_factor = discount_factor
        self.steps = 0

    def reward(self, reward):
        return reward * pow(self.discount_factor, self.steps)

    def step(self, action):
        self.steps += 1
        return super().step(action)

    def reset(self, **kwargs):
        self.steps = 0
        return super().reset(**kwargs)
