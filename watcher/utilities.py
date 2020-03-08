import math
import os
from collections import deque
from math import sqrt

import cv2
import numpy as np
import tensorflow as tf
import typing
from PIL import Image

from from_internet_or_for_from_internet import PNP_solver


def path_change_decorator(func: typing.Callable) -> typing.Callable:
    """Decorate a function to make it easy to take resources from local directory of the function

    :param func:
    :return:
    """

    def result_func(*args, **kwargs) -> typing.Any:
        """Change current directory to the function's module one, apply func and make dir back

        :param args:
        :param kwargs:
        :return:
        """
        old_path = os.getcwd()
        # noinspection PyUnresolvedReferences
        module_path = func.__module__.split(".")[0:-1]
        watcher_path = os.path.join(*(__file__.split("/")[:-1]))
        new_path = os.path.join(watcher_path, *module_path)
        os.chdir("/" + new_path)
        result = func(*args, **kwargs)
        os.chdir(old_path)
        return result

    return result_func


def normalize(v: np.ndarray):
    """Convert vector to unit vector of same direction

    :param v: a vector to normalize
    :return: unit vector of same direction
    """
    norm = 0
    for vi in v:
        norm += vi * vi
    norm = sqrt(norm)

    if norm == 0:
        raise ArithmeticError("Vector of length 0")
    return v / norm


def get_world_to_camera_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Get world to camera coordinate system translation matrix

    :param rotation: shape [3], Rodrigues rotation vector
    :param translation: shape [3]
    :return: shape [3, 4], world to camera matrix
    """
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[0:3, 0:3], _ = cv2.Rodrigues(rotation)
    matrix[:, 3] = translation.reshape((3,))
    return matrix


def get_world_to_camera_projection_matrix(solver: PNP_solver.PoseEstimator,
                                          rotation: np.ndarray,
                                          translation: np.ndarray,
                                          is_vector_translation: bool = False) -> np.ndarray:
    """Construct world to camera picture projection matrix

    Matrix construction is a bit different for vector and for point translation as vector
    doesn't use translation

    :param solver:
    :param rotation: shape [3]
    :param translation: shape [3]
    :param is_vector_translation: whether
    :return: shape [3, 4], projection matrix
    """
    if is_vector_translation:
        translation = np.zeros((3,))

    matrix = get_world_to_camera_matrix(rotation, translation)
    return np.matmul(solver.camera_matrix, matrix)


FIRST_SMOOTH_TIME = 1
SECOND_SMOOTH_TIME = 3


def smooth_n_cut(time_list: typing.Deque[typing.List], time_now: float) -> typing.List:
    """Remove all that is older than SECOND_SMOOTH_TIME and smooth time series

    :param time_list: time series, deque of lists [..., time]
    :param time_now: current time
    :return: smoothed list [...]
    """
    while abs(time_list[0][-1] - time_now) > SECOND_SMOOTH_TIME:
        time_list.popleft()
    return smooth_func(time_list, time_now, FIRST_SMOOTH_TIME, SECOND_SMOOTH_TIME)


def smooth_func(time_list: typing.Deque[typing.List], time_now: float,
                first_part_time: float = FIRST_SMOOTH_TIME,
                second_part_time: float = SECOND_SMOOTH_TIME) -> typing.List:
    """Smooth time series list values

    :param time_list: time series, deque of lists [..., time]
    :param time_now: current time
    :param first_part_time: same weight
    :param second_part_time: descending weight
    :return: smoothed list [...]
    """
    if not isinstance(time_list, np.ndarray):
        time_list = np.array(time_list)

    ave = np.zeros((len(time_list[0]) - 1), dtype=np.float32)
    total_weight = 0.

    for cur_time in time_list:
        time = cur_time[-1]
        value = cur_time[0:len(cur_time) - 1]

        if abs(time_now - time) < first_part_time:
            weight = 1
        elif abs(time_now - time) < second_part_time:
            time -= first_part_time
            tang = math.tan(1 / (second_part_time - first_part_time))
            weight = 1. - time * tang
        else:
            break

        ave += value * weight
        total_weight += weight

    if total_weight == 0:
        total_weight = 1

    ave = ave / total_weight
    return ave


