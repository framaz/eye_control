import collections

import numpy as np
from matplotlib import pyplot as plt

import utilities
from camera_holders import camera_system_factory, seen_to_screen

SHOW_EYE_HISTORY_AND_BOARDERS = False


class Eye:
    def __init__(self, eye_type):
        assert eye_type == 'l' or eye_type == 'r'
        self.corner_vectors = np.zeros((len(camera_system_factory.corner_dict), 3))
        self.translator = None
        self.corner_points = np.zeros((len(camera_system_factory.corner_dict), 2))
        self.history = collections.deque()
        self.screen = None
        self.tmp_value = 0
        self.eye_type = eye_type

    def create_translator(self, screen, eye_center):
        self.screen = screen
        self.corner_vectors = np.array(self.corner_vectors)
        for i in range(self.corner_vectors.shape[0]):
            self.corner_points[i] = utilities.get_pixel(self.corner_vectors[i], eye_center)
        self.translator = seen_to_screen.SeenToScreenTranslator(self.corner_points)
        self.translator = self.translator

    def add_screen(self, screen):
        self.screen = screen

    def add_history(self, vector, time_now, eye_center):
        # TODO normal eye center detection
        vector = utilities.normalize(vector)
        self.tmp_value += 1
        if self.tmp_value >= 10 and SHOW_EYE_HISTORY_AND_BOARDERS:
            plt.plot(self.corner_points[:, 0], self.corner_points[:, 1], )
            np_history = np.array(self.history)
            plt.plot(np_history[:, 0], np_history[:, 1], )
            plt.show()
            self.tmp_value = 0
        res_pixel = utilities.get_pixel(vector, eye_center)
        point = res_pixel
        self.history.append([point[0], point[1], time_now])
        seen_point = utilities.smooth_n_cut(self.history, time_now)
        screen_point = self.translator.seen_to_screen(seen_point)
        return screen_point