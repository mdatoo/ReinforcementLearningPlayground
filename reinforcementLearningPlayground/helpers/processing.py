from typing import List

import cv2
import numpy as np


def mean(values: List[float]) -> float:
    if len(values) == 0:
        return 0
    return np.mean(values)


def normalize(arr: np.ndarray) -> np.ndarray:
    if np.count_nonzero(arr) == 0:
        return np.ones_like(arr) / arr.size
    return arr / np.sum(arr)


def resize_image(image: np.ndarray, new_shape: (int, int)) -> np.ndarray:
    return cv2.resize(image, new_shape)
