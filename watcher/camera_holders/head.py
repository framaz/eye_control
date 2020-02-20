import collections

import numpy as np

import utilities
from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from utilities import get_world_to_camera_matrix


class Head:
    def __init__(self, starting_rotation_vector, starting_translation: np.ndarray, solver: pnp_solver):
        self.solver = solver
        self.start_rotation = starting_rotation_vector
        self.start_translation = starting_translation
        self.rotation_history = collections.deque()
        self.translation_history = collections.deque()

    def add_history(self, np_points, time_now):
        rotation, translation = self.solver.solve_pose(np_points)
        rotation = rotation.reshape((3,))
        translation = translation.reshape((3,))
        self.rotation_history.append([*rotation, time_now])
        self.translation_history.append([*translation, time_now])
        smoothed_rotation = utilities.smooth_n_cut(self.rotation_history, time_now)
        smoothed_translation = utilities.smooth_n_cut(self.translation_history, time_now)
        return smoothed_rotation, smoothed_translation

    def translate_vector_to_first_position(self, starting_rotation_vector, starting_translation):
        to_new_world = get_world_to_camera_matrix(self.solver, starting_rotation_vector, starting_translation,
                                                  is_vector_translation=True)