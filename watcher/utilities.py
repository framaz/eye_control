import math
import os
from math import sqrt

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def path_change_decorator(func):
    def kek(*args, **kwargs):
        old_path = os.getcwd()
        module_path = func.__module__.split(".")[0:-1]
        watcher_path = os.path.join(*(__file__.split("/")[:-1]))
        new_path = os.path.join(watcher_path, *module_path)
        os.chdir("/"+new_path)
        result = func(*args, **kwargs)
        os.chdir(old_path)
        return result
    return kek


def normalize(v):
    norm = 0
    for vi in v:
        norm += vi * vi
    norm = sqrt(norm)
    if norm == 0:
        return v
    return v / norm


def vector_to_camera_coordinate_system(rotation, translation):
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[0:3, 0:3] = rotation_matrix
    matrix[:, 3] = translation.reshape((3,))
    return matrix


def get_world_to_camera_matrix(solver, starting_rotation_vector, starting_translation, is_vector_translation=False):
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[0:3, 0:3], _ = cv2.Rodrigues(starting_rotation_vector)
    if not is_vector_translation:
        matrix[:, 3] = starting_translation.reshape((3,))
    return np.matmul(solver.camera_matrix, matrix)


FIRST_SMOOTH_TIME = 1
SECOND_SMOOTH_TIME = 3

def smooth_n_cut(time_list, time_now):
    while abs(time_list[0][-1] - time_now) > SECOND_SMOOTH_TIME:
        time_list.popleft()
    return smooth_func(time_list, time_now, FIRST_SMOOTH_TIME, SECOND_SMOOTH_TIME)


def smooth_func(time_list, time_now, first_part_time=FIRST_SMOOTH_TIME, second_part_time=SECOND_SMOOTH_TIME):
    if not isinstance(time_list, np.ndarray):
        time_list = np.array(time_list)
    ave = np.zeros((len(time_list[0]) - 1), dtype=np.float32)
    total_weight = 0.
    for cur_time in time_list:
        time = cur_time[-1]
        value = cur_time[0:len(cur_time) - 1]
        weight = 0
        if abs(time_now - time) < first_part_time:
            weight = 1
        elif abs(time_now - time) < second_part_time:
            time -= first_part_time
            tang = math.tan(1 / (second_part_time - first_part_time))
            weight = 1. - (time) * tang
        else:
            break
        ave += value * weight
        total_weight += weight
    if total_weight == 0:
        total_weight = 1
    ave = ave / total_weight
    return ave


def pack_to_one_image(*args):
    sizes = list(map(lambda image: image.size, args))
    heigth = max(map(lambda size: size[1], sizes))
    length = sum(map(lambda size: size[0], sizes))
    mode = args[0].mode
    pointer = 0
    result = Image.new(mode, (length, heigth))
    for pic in args:
        result.paste(pic, (pointer, 0))
        pointer += pic.width
    return result


def get_pixel(tens, a, b, c, d,
              width=0.54, heigth=0.30375,
              pixel_width=1920, pixel_heigth=1080):
    """k = -d / (a * x + b * y + c * z)
    print(x[0].item())
    x = k * x
    y = k * y
    z = k * z
    pixel_x = x / width * pixel_width
    pixel_y = y / heigth * pixel_heigth
    return pixel_x, pixel_y """
    fl = np.array([a, b, c])
    ratios = np.array([width * pixel_width, heigth * pixel_heigth])
    bottom = tf.math.reduce_sum(tens * fl, 1)
    k = d / (tf.math.reduce_sum(tens * fl, 1))
    k = tf.convert_to_tensor([k, k, k])
    k = tf.transpose(k)
    tens = tens * k
    tens = tens[:, 0:2]
    tens = tens * ratios
    return tens