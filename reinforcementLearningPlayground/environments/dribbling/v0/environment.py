from collections import namedtuple
from random import randint
from typing import Optional

import cv2
import numpy as np
import gym
import pymunk

from ....helpers import DrawOptions

ACTIONS = 4
BASE_REWARD = 1
CHANNELS = 3

COLLISION_TYPES = {
    "ball": 1,
    "player": 2,
    "goal": 3,
    "border": 4
}

Objects = namedtuple("Objects", ["player", "ball", "goal"])


class Environment(gym.Env):
    def __init__(self, height: int, width: int):
        super().__init__()

        self._height = height
        self._width = width
        self._observation_space = gym.spaces.Box(low=np.zeros(self.observation_shape),
                                                 high=np.ones(self.observation_shape),
                                                 dtype=float)
        self._action_space = gym.spaces.Box(low=np.zeros(ACTIONS),
                                            high=np.ones(ACTIONS),
                                            dtype=int)
        self._reward_range = (0, BASE_REWARD)
        self.space = None
        self.objects = None
        self.print_options = None
        self.done = False
        self.steps = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def observation_shape(self) -> (int, int, int):
        return self.height, self.width, CHANNELS

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def reward_range(self) -> (int, int):
        return self._reward_range

    def reset(self) -> np.ndarray:
        self.space = pymunk.Space()
        self.print_options = DrawOptions(self.width, self.height)

        borders = [
            pymunk.Segment(self.space.static_body, (0, 0), (self.width, 0), 5),
            pymunk.Segment(self.space.static_body, (self.width, 0), (self.width, self.height), 5),
            pymunk.Segment(self.space.static_body, (self.width, self.height), (0, self.height), 5),
            pymunk.Segment(self.space.static_body, (0, self.height), (0, 0), 5)
        ]
        for border in borders:
            border.elasticity = 1.0
            border.collision_type = COLLISION_TYPES["border"]
        self.space.add(*borders)

        player = pymunk.Body()
        player.position = (self.width // 5, self.height // 2)
        player_poly = pymunk.Poly.create_box(player, (15, 15))
        player_poly.mass = 10
        player_poly.elasticity = 1.0
        player_poly.collision_type = COLLISION_TYPES["player"]
        player_poly.filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 1)
        self.space.add(player, player_poly)

        ball = pymunk.Body()
        ball.position = (self.width // 2, self.height // 2)
        ball_poly = pymunk.Circle(ball, 3)
        ball_poly.mass = 2
        ball_poly.elasticity = 1.0
        ball_poly.collision_type = COLLISION_TYPES["ball"]
        self.space.add(ball, ball_poly)

        goal_position = (randint(round(self.width * 0.1), round(self.width * 0.9)),
                         randint(round(self.height * 0.1), round(self.height * 0.9)))
        goal = pymunk.Circle(self.space.static_body, 3, goal_position)
        goal.collision_type = COLLISION_TYPES["goal"]
        goal.filter = pymunk.ShapeFilter(categories=1)
        self.space.add(goal)

        self.objects = Objects(player, ball, goal)

        def set_done(_, __, ___):
            self.done = True
            return True

        h = self.space.add_collision_handler(COLLISION_TYPES["ball"], COLLISION_TYPES["goal"])
        h.begin = set_done

        self.done = False
        self.steps = 0

        return self.render("rgb_array")

    def render(self, mode="human") -> Optional[np.ndarray]:
        assert mode in ["human", "rgb_array"], f"Invalid mode {mode}, must be either 'human' or 'rgb_array'"

        self.print_options.reset()
        self.space.debug_draw(self.print_options)

        if mode == "human":
            cv2.imshow("Game", self.print_options.image)
            cv2.waitKey(1)
            return None
        else:
            return self.print_options.image

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        assert action in self.action_space, f"Invalid action {action}, must be np.ndarray of integers with elements " \
                                            f"between 0 and 1 of shape ({ACTIONS},)"
        assert not self.done, "Called step when environment is already done"

        self.objects.player.velocity = (action[2] - action[0], action[3] - action[1])
        self.space.step(1)
        self.steps += 1

        if self.done:
            return self.render("rgb_array"), BASE_REWARD, True, {}
        else:
            return self.render("rgb_array"), 0.0, False, {}

    def close(self) -> None:
        cv2.destroyAllWindows()
