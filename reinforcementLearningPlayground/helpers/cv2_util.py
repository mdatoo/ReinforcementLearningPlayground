from typing import Sequence

import cv2
import numpy as np
from pymunk import SpaceDebugDrawOptions, Vec2d
from pymunk.space_debug_draw_options import SpaceDebugColor

CHANNELS = 3


class DrawOptions(SpaceDebugDrawOptions):
    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self._width = width
        self._height = height
        self._image = np.full((height, width, CHANNELS), 255, dtype=np.uint8)

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, width: int) -> None:
        self._width = width
        self.reset()

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, height: int) -> None:
        self._height = height
        self.reset()

    def reset(self):
        self._image = np.full((self.height, self.width, CHANNELS), 255, dtype=np.uint8)

    @property
    def image(self) -> np.ndarray:
        return self._image

    def draw_circle(
            self,
            pos: Vec2d,
            angle: float,
            radius: float,
            outline_color: SpaceDebugColor,
            fill_color: SpaceDebugColor
    ) -> None:
        self._image = cv2.circle(self._image, (round(pos.x), round(pos.y)), round(radius), fill_color.as_int(), -1)
        self._image = cv2.circle(self._image, (round(pos.x), round(pos.y)), round(radius), outline_color.as_int())

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        self._image = cv2.line(self._image, (round(a.x), round(a.y)), (round(b.x), round(b.y)), color.as_int())

    def draw_fat_segment(
        self,
        a: Vec2d,
        b: Vec2d,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        self._image = cv2.line(self._image, (round(a.x), round(a.y)), (round(b.x), round(b.y)), fill_color.as_int(),
                               round(radius))

    def draw_polygon(
        self,
        verts: Sequence[Vec2d],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        verts_converted = np.array([[round(vert.x), round(vert.y)] for vert in verts])

        self._image = cv2.fillPoly(self._image, [verts_converted], fill_color)
        self._image = cv2.polylines(self._image, [verts_converted], True, outline_color, round(radius))

    def draw_dot(self, size: float, pos: Vec2d, color: SpaceDebugColor) -> None:
        self._image = cv2.circle(self._image, (round(pos.x), round(pos.y)), round(size), color)
