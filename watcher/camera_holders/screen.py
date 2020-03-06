from __future__ import annotations

import numpy as np
import typing
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utilities import vector_to_camera_coordinate_system
import from_internet_or_for_from_internet.PNP_solver as pnp_solver


class Screen:
    """Used to incorporate all screen plane parameters a, b, c, d and methods for plane cross

    :ivar a:
    :ivar b:
    :ivar c:
    :ivar d:
    """

    def __init__(self, r_eye: Eye, l_eye: Eye,
                 solver: pnp_solver.PoseEstimator,
                 head_rotation: np.ndarray, head_translation: np.ndarray):
        """Construct object

        Also draws a plot

        :param r_eye: right eye
        :type r_eye: camera_holders.Eye
        :param l_eye: left eye
        :type l_eye: camera_holders.Eye
        :param solver:
        :param head_rotation: shape [3], rotation vector
        :param head_translation: shape [3]
        """
        assert len(l_eye.corner_points) == 5
        assert len(r_eye.corner_points) == 5

        self.solver = solver

        world_to_camera = vector_to_camera_coordinate_system(head_rotation, head_translation)
        kek = []

        # translate all face points to camera coordinate system
        for point in solver.model_points_68:
            kek.append(world_to_camera @ [*point, 1])
        kek = np.array(kek).transpose()

        # get eyes centers
        left_eye = sum(self.solver.model_points_68[36:41]) / 6
        right_eye = sum(self.solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)

        pairs_list = []
        for i in range(len(l_eye.corner_vectors)):
            pair = {'left': [left_eye, l_eye.corner_vectors[i]], 'right': [right_eye, r_eye.corner_vectors[i]]}
            pairs_list.append(pair)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(*kek)

        colors = ['#ff0000', '#00ff00', '#0000ff', '#000000', '#ffffff']
        length = 1800
        for screen_point, color in zip(pairs_list, colors):
            arr = np.array(screen_point['left'])
            arr[1] = arr[1] * length + arr[0]
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=color)
            arr = np.array(screen_point['right'])
            arr[1] = arr[1] * length + arr[0]
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=color)

        # method for plane building by skew lines, now not used
        # self.a, self.b, self.c, self.d = plane_by_eye_vectors.get_plane_by_eye_vectors(pairs_list)

        # surface plane with avg(gaze_vectors) being as normal vector
        normal_vector = sum(l_eye.corner_vectors) + sum(r_eye.corner_vectors)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        [a, b, c] = normal_vector
        eye_middle = (left_eye + right_eye) / 2
        screen_point = eye_middle + length * normal_vector
        d = - normal_vector @ screen_point
        self.a, self.b, self.c, self.d = a, b, c, d

        def draw_plot(plane, axis, min_x=-200, min_y=-200, max_x=205, max_y=205):
            a, b, c, d = plane
            xx, yy = np.meshgrid(range(min_x, max_x + 5, max_x - min_x), range(min_y, max_y + 5, max_y - min_y))
            z = -(a * xx + b * yy + d) / c
            axis.plot_wireframe(xx, yy, z)

        draw_plot([a, b, c, d], ax)
        # plt.show()

    def get_2d_cross_position(self, gaze_vector: np.ndarray,
                              eye_point: np.ndarray = None) -> typing.List[float]:
        """Get the cross point of the plane and gaze_vector

        It just cuts the z position of 3d cross position so if the plane is perpendicular to x or y then
        the func will fail.

        :param gaze_vector: shape [3], in camera coordinate system
        :param eye_point: shape [3], eye center in camera coordinate system
        :return: shape [2], x and y positions of
        """
        if eye_point is None:
            eye_point = [0, 0, 0]

        [xv, yv, zv] = gaze_vector
        [x0, y0, z0] = eye_point

        # to make it easier
        a = self.a
        b = self.b
        c = self.c
        d = self.d

        t = - (a * x0 + b * y0 + c * z0 + d) / (a * xv + b * yv + c * zv)

        x = t * xv + x0
        y = t * yv + y0
        # z = t * zv + z0

        return [x, y]
