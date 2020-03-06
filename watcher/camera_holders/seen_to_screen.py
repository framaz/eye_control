import copy
import math

import numpy as np
import typing


def get_line_equation(start_point: np.ndarray,
                      end_point: np.ndarray) -> typing.Tuple[float, float, float]:
    """Get line equation in form ax + by + c = 0 by 2 points

    :param start_point: shape [2]
    :param end_point: shape[2]
    """
    [x1, y1] = start_point
    [x2, y2] = end_point

    x = x2 - x1
    y = y2 - y1

    if x == 0 and y == 0:
        raise ArithmeticError
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


def get_distance_to_line(line_start: np.ndarray, line_end: np.ndarray, p_target: np.ndarray):
    """Get distance from p_target point to line described by line_start and line_end in 2d space

    :param line_start: shape [2]
    :param line_end: shape [2]
    :param p_target: shape [2]
    """
    a, b, c = get_line_equation(line_start, line_end)
    [x, y] = p_target

    return abs(a * x + b * y + c) / math.sqrt(x * x + y * y)


def get_screen_point_array(width: float, height: float):
    """Get screen points(corners) in pixels from normalized points_in_square

    :param width: screen width
    :param height: screen height
    :return:
    """
    points = copy.deepcopy(points_in_square)
    for i in range(len(points_in_square)):
        points[i] = points[i][0] * width, points[i][1] * height

    result = list_points_to_triangle(points)
    return np.array(result, dtype=np.float32)


def list_points_to_triangle(points: typing.List[float]):
    """Forms triangles from a list of points

    Triangle order is in triangles list

    :param points: list of triangles
    """
    result = []

    for triangle in triangles:
        points_of_triangle = []
        for point_num in triangle:
            points_of_triangle.append(points[point_num])
        result.append(points_of_triangle)
    return np.array(result, dtype=np.float32)


def create_basis_translation_matrixes(triangles_list: np.ndarray, inv: bool = True):
    """Create a list of all translation matrixes from triangles in real world to screen ones

    The idea of the method:
    Every point inside a triangle is a linear combination of two vectors-sides
    of triangle from one point. The vectors and the point are the coordinate system of triangle.
    If two vectors-sides of different triangles are matched and a point in first triangle has
    coordinates x and y in the coord system of first triangle then corresponding point in second
    triangle has same coordinates x and y but in coordinate system of second triangle

    :param triangles_list:shape [n, 3, 2] list of triangles in real world
    :param inv: whether to inv resulting translation matrixes
    :return:
    """
    translations = []
    for i in range(len(triangles_list)):
        translator_matrix = np.zeros((2, 2), dtype=np.float32)
        triangle_seen = triangles_list[i]

        # Create translation matrix from starting basis to triangle basis
        for num in range(2):
            translator_matrix[num, 0] = triangle_seen[num + 1, 0] - triangle_seen[0, 0]
            translator_matrix[num, 1] = triangle_seen[num + 1, 1] - triangle_seen[0, 1]
        translator_matrix = translator_matrix.transpose()

        # Depending on direction of application of translation matrix we may want to inv it now
        if inv:
            try:
                translator_matrix = np.linalg.inv(translator_matrix)
            except np.linalg.LinAlgError:
                raise
        translations.append(translator_matrix)
    return translations


class SeenToScreenTranslator:
    """A class for translating points from basic coordinate system to screen pixels

    :ivar _width: float
    :ivar _heigth: float
    :ivar _triangles_on_screen: np.ndarray, shape [n, 3, 2]
    :ivar _triangles_seen: np.ndarray, shape [n, 3, 2], triangles as seen in camera coordinate sys
    :ivar _to_seen_vector_translation_matrixes:
    :ivar _to_screen_vector_translation_matrixes:
    """

    def __init__(self, point_list):
        self._width = 1920
        self._heigth = 1080
        self._triangles_on_screen = get_screen_point_array(self._width, self._heigth)
        self._triangles_seen = list_points_to_triangle(point_list)

        assert len(self._triangles_on_screen) == len(self._triangles_seen)
        self._to_seen_vector_translation_matrixes = create_basis_translation_matrixes(self._triangles_seen)
        self._to_screen_vector_translation_matrixes = create_basis_translation_matrixes(self._triangles_on_screen,
                                                                                        inv=False)

    def check_triangle_accessory(self, point: np.ndarray):
        """Find in which triangle is the point

        For each side of triangle it is checked whether point and third vertex of triangle are
        on the same side. If its true then the point is in triangle

        :param point: shape [2]
        :return: triangle number
        """
        if isinstance(point, list):
            point = np.array(point, np.float32)

        for i in range(len(self._triangles_seen)):
            triangle_seen = self._triangles_seen[i]
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

        # if the point isnt in any triangle then the nearest is finded
        for i in range(len(self._triangles_seen)):
            triangle_seen = self._triangles_seen[i]
            distances = []
            for j in range(3):
                distance = get_distance_to_line(triangle_seen[j], triangle_seen[(j + 1) % 3], point)
                distances.append(distance)
            if min_value > min(distances):
                triangle_target = i
                min_value = min(distances)
        return triangle_target

    def seen_to_screen(self, point: np.ndarray):
        """Translate a point from outer system to screen pixel

        :param point: shape [2]
        :return:
        """
        target = self.check_triangle_accessory(point)

        point = point - self._triangles_seen[target, 0]
        coords_of_vect = np.matmul(self._to_seen_vector_translation_matrixes[target], point)

        point = np.matmul(self._to_screen_vector_translation_matrixes[target], coords_of_vect)
        point += self._triangles_on_screen[target, 0]
        return point


triangles = [[0, 1, 4],
             [1, 2, 4],
             [2, 3, 4],
             [3, 0, 4]
             ]
points_in_square = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
