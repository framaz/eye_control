import numpy as np
from PIL import Image

eye_padding_ratio = 1.2


def slice_eye(img, parts):
    left = min(parts[:, 0])
    top = min(parts[:, 1])
    right = max(parts[:, 0])
    bottom = max(parts[:, 1])
    size, middle_x, middle_y = 0, 0, 0
    if right - left >= top - bottom:
        size = right - left
    else:
        size = top - bottom
    if size % 2 == 1:
        size += 1
    size += 6
    ratio = 60. / 36.
    middle_y = int((top + bottom) / 2)
    middle_x = int((left + right) / 2)
    bottom = int(middle_y + size / ratio / 2 * eye_padding_ratio)
    top = int(middle_y - size / ratio / 2 * eye_padding_ratio)
    left = int(middle_x - size / 2 * eye_padding_ratio)
    right = int(middle_x + size / 2 * eye_padding_ratio)
    result = img[top:bottom, left:right]
    return result, left, top


def find_eye_middle(nparray):
    eye_borders = nparray[(36, 39, 42, 45), :].astype(dtype=np.float32)
    a = sum(eye_borders[:, 0])
    b = sum(eye_borders[:, 1])
    mid_x = a / 4
    mid_y = b / 4
    """[x1, y1] = eye_borders[0]
    [x2, y2] = eye_borders[3]
    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    cumul = 0
    for [x, y] in eye_borders:
        cumul += ((k * x + b) - y) ** 2
    return (1, k), np.array([mid_x, mid_y])"""
    eye_borders -= np.array([mid_x, mid_y] * 4).reshape((4, 2))
    x = eye_borders[:, 0]
    y = eye_borders[:, 1]
    k = sum(x * y) / sum(x * x)
    vect = np.array([1, k])
    return vect, np.array([mid_x, mid_y])


PADDING = 10


def process_eye(face, np_points):
    eye, left, top = slice_eye(np.array(face), np_points)
    eye = Image.fromarray(eye)
    eye = eye.resize((60, 36))
    np_eye = np.array(eye)
    np_eye = np_eye.reshape((36, 60, 1))
    np_eye = np_eye.astype(dtype=np.float32)
    np_eye /= 255.
    return np_eye, left, top
