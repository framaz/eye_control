import collections

import numpy as np

import utilities
from from_internet_or_for_from_internet import PNP_solver as PNP_solver


class Head:
    """Class for head translation and rotation

    :ivar solver: (pnp_solver.PoseEstimator)
    :ivar _rotation_history"""
    def __init__(self, solver: PNP_solver.PoseEstimator):
        """Constructs object"""
        self._solver = solver
        self._rotation_history = collections.deque()
        self._translation_history = collections.deque()

    def get_smoothed_position(self, np_points: np.ndarray, time_now: float):
        """Add current head pos to history and get smoothed head pos

        :param np_points:
        :param time_now:
        :return:
        """
        rotation, translation = self._solver.solve_pose(np_points)

        rotation = rotation.reshape((3,))
        translation = translation.reshape((3,))

        self._rotation_history.append([*rotation, time_now])
        self._translation_history.append([*translation, time_now])

        smoothed_rotation = utilities.smooth_n_cut(self._rotation_history, time_now)
        smoothed_translation = utilities.smooth_n_cut(self._translation_history, time_now)
        return smoothed_rotation, smoothed_translation
