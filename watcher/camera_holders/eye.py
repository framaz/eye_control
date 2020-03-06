from __future__ import annotations

import collections

import numpy as np
from matplotlib import pyplot as plt

import utilities
from . import camera_system_factory, seen_to_screen

SHOW_EYE_HISTORY_AND_BOARDERS = False


class Eye:
    """Class for one eye

    :ivar corner_vectors: (np.array of shape[n, 3]) used to store 3d gaze vectors of corners
    :ivar _translator: (seen_to_screen.SeenToScreenTranslator)
                       used to translate point from on screen 3d plane to pixel on screen
    :ivar corner_points: (np.array of shape[n, 2])
                         used to store pixel positions corresponding to 3d gaze vectors of corners
    :ivar _history: (collections.deque, format [pixel_x, pixel_y, time]) remembers pixel history
    :ivar _screen: used to get cross of gaze and screen plane
    """

    def __init__(self):
        """Construct Eye

        Every object value is empty
        """
        self.corner_vectors = np.zeros((len(camera_system_factory.corner_dict), 3))
        self._translator = None
        self.corner_points = np.zeros((len(camera_system_factory.corner_dict), 2))
        self._history = collections.deque()
        self._screen = None

    def create_translator(self, screen: Screen, eye_center: np.ndarray) -> None:
        """Create translator by corner vectors and set screen

        :param screen: screen to use for the eye
        :param eye_center: eye center in camera coordinate system
        :return:
        """
        self._screen = screen
        self.corner_vectors = np.array(self.corner_vectors)

        for i in range(self.corner_vectors.shape[0]):
            self.corner_points[i] = screen.get_2d_cross_position(self.corner_vectors[i], eye_center)
        self._translator = seen_to_screen.SeenToScreenTranslator(self.corner_points)

    def get_screen_point(self,
                         vector: np.ndarray,
                         time_now: int,
                         eye_center: np.ndarray) -> np.ndarray:
        """Remember and smooth current gaze vector, get pixel on screen

        :param vector: shape [3], gaze vector
        :param time_now: current time (time.time())
        :param eye_center: shape [3]
        :return: shape [2], point on screen in pixels
        """
        vector = utilities.normalize(vector)

        point = self._screen.get_2d_cross_position(vector, eye_center)

        self._history.append([point[0], point[1], time_now])
        seen_point = utilities.smooth_n_cut(self._history, time_now)

        screen_point = self._translator.seen_to_screen(seen_point)
        return screen_point
