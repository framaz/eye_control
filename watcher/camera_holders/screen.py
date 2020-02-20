import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities import vector_to_camera_coordinate_system
from . import Eye


class Screen:
    def __init__(self, l_eye: Eye, r_eye: Eye, solver, head_rotation, head_translation):
        # TODO non-dumb screen surface creation
        assert len(l_eye.corner_points) == 5
        assert len(r_eye.corner_points) == 5
        self.solver = solver
        pairs_list = []
        kek = []
        world_to_camera = vector_to_camera_coordinate_system(head_rotation, head_translation)
        for point in solver.model_points_68:
            kek.append(world_to_camera @ [*point, 1])
        kek = np.array(kek).transpose()
        left_eye = sum(self.solver.model_points_68[36:41]) / 6
        right_eye = sum(self.solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)
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
     #   self.a, self.b, self.c, self.d = plane_by_eye_vectors.get_plane_by_eye_vectors(pairs_list)
        self.a, self.b, self.c, self.d = 0, 0, 1, -450

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
        #plt.show()
        self.width = 0.54
        self.heigth = 0.30375
        self.pixel_width = 1920
        self.pixel_heigth = 1080

    def get_pixel(self, gaze_vector, eye_point=None):
        if eye_point is None:
            eye_point = [0, 0, 0]
        [xv, yv, zv] = gaze_vector
        [x0, y0, z0] = eye_point
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        t = - (a * x0 + b * y0 + c * z0 + d) / (a * xv + b * yv + c * zv)
        x = t * xv + x0
        y = t * yv + y0
        z = t * zv + z0

        pixel_x = x
        pixel_y = y
        return [pixel_x, pixel_y]