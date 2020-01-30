import copy
import math

import numpy as np


def get_linear_equation(p1, p2):
    [x1, y1] = p1
    [x2, y2] = p2
    x = x2 - x1
    y = y2 - y1
    a = 0
    b = 0
    if x == 0 and y == 0:
        return None
    if x == 0:
        a = 1
        b = 0
    elif y == 0:
        a = 0
        b = 1
    else:
        b = 1
        a = -b * y / x
    c = -a * x1 - b * y1
    return a, b, c


def get_distance_to_line(p1, p2, p_target):
    a, b, c = get_linear_equation(p1, p2)
    [x, y] = p_target
    return abs(a * x + b * y + c) / math.sqrt(x * x + y * y)


def get_screen_point_array(width, heigth):
    points = copy.deepcopy(points_in_square)
    for i in range(len(points_in_square)):
        points[i] = points[i][0] * width, points[i][1] * heigth
    result = list_points_to_triangle(points)
    return np.array(result, dtype=np.float32)


def list_points_to_triangle(points_in_square):
    result = []
    for triangle in triangles:
        points_of_triangle = []
        for point_num in triangle:
            points_of_triangle.append(points_in_square[point_num])
        result.append(points_of_triangle)
    return np.array(result, dtype=np.float32)


def create_basis_translation_matrixes(triangles, inv=True):
    translations = []
    for i in range(len(triangles)):
        translator_matrix = np.zeros((2, 2), dtype=np.float32)
        triangle_seen = triangles[i]
        for num in range(2):
            translator_matrix[num, 0] = triangle_seen[num + 1, 0] - triangle_seen[0, 0]
            translator_matrix[num, 1] = triangle_seen[num + 1, 1] - triangle_seen[0, 1]
        translator_matrix = translator_matrix.transpose()
        if inv:
            try:
                translator_matrix = np.linalg.inv(translator_matrix)
            except np.linalg.LinAlgError:

                raise
        translations.append(translator_matrix)
    return translations


class SeenToScreenTranslator:
    def __init__(self, point_list):
        # TODO autodetect in pixels
        self.width = 1920
        self.heigth = 1080
        self.triangles_on_screen = get_screen_point_array(self.width, self.heigth)
        self.triangles_seen = list_points_to_triangle(point_list)

        assert len(self.triangles_on_screen) == len(self.triangles_seen)
        self.to_seen_vector_translation_matrixes = create_basis_translation_matrixes(self.triangles_seen)
        self.to_screen_vector_translation_matrixes = create_basis_translation_matrixes(self.triangles_on_screen,
                                                                                       inv=False)

    #       kek = np.matmul(self.to_seen_vector_translation_matrixes[0], [1, 2])
    #        lol = np.matmul(self.to_screen_vector_translation_matrixes[0], kek, )

    def check_triangle_accessory(self, point):
        if isinstance(point, list):
            point = np.array(point, np.float32)
        for i in range(len(self.triangles_seen)):
            triangle_seen = self.triangles_seen[i]
            crosses = []
            for j in range(3):
                cur_cross = np.cross(point - triangle_seen[j], triangle_seen[(j + 1) % 3] - triangle_seen[j])
                crosses.append(cur_cross)
            if crosses[0] >= 0 and crosses[1] >= 0 and crosses[2] >= 0:
                return i
            if crosses[0] <= 0 and crosses[1] <= 0 and crosses[2] <= 0:
                return i
        min_value = np.inf
        triangle_target = -1
        for i in range(len(self.triangles_seen)):
            triangle_seen = self.triangles_seen[i]
            distances = []
            for j in range(3):
                distance = get_distance_to_line(triangle_seen[j], triangle_seen[(j + 1) % 3], point)
                distances.append(distance)
            if min_value > min(distances):
                triangle_target = i
                min_value = min(distances)
        return triangle_target

    def seen_to_screen(self, point):
        target = self.check_triangle_accessory(point)
        point = point - self.triangles_seen[target, 0]
        coords_of_vect = np.matmul(self.to_seen_vector_translation_matrixes[target], point)
        point = np.matmul(self.to_screen_vector_translation_matrixes[target], coords_of_vect)
        point += self.triangles_on_screen[target, 0]
        return point


triangles = [[0, 1, 4],
             [1, 2, 4],
             [2, 3, 4],
             [3, 0, 4]
             ]
points_in_square = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]