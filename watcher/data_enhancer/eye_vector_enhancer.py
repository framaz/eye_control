import copy

import cv2
import numpy as np
from PIL import ImageDraw

from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from utilities import normalize
from .head_axis_enhancer import HeadAxisEnhancer


class EyeVectorEnhancer(HeadAxisEnhancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, pic, np_points, eye_vector_left, eye_vector_right):
        pic, output = super().process(pic, np_points, draw_points=True)
        solver = pnp_solver.PoseEstimator((720, 1080))
        drawer = ImageDraw.Draw(pic)
        rotation, translation = solver.solve_pose(np_points)
        back_matrix = self.get_triangle_face_to_just_face_matrix(solver)
        triangle_matrix, _ = cv2.Rodrigues(back_matrix)
        matrix, _ = cv2.Rodrigues(rotation)
        res = np.matmul(matrix, back_matrix)
        res, _ = cv2.Rodrigues(res)
        #pic = solver.draw_axes(pic, res, translation)

        self.draw_eye_vector(drawer, eye_vector_left, solver, rotation, translation, eye_pos="l")
        self.draw_eye_vector(drawer, eye_vector_right, solver, rotation, translation, eye_pos="r")
        return pic, output

    def get_triangle_face_to_just_face_matrix(self, solver):
        all_points = solver.model_points_68
        l_eye_middle = (all_points[36] + all_points[39]) / 2
        r_eye_middle = (all_points[42] + all_points[45]) / 2
        mouth_middle = (all_points[48] + all_points[54]) / 2
        x_axis = r_eye_middle - l_eye_middle
        x_axis = normalize(x_axis)
        # drawer.ellipse([*(x_axistmp-2), *(x_axistmp+2)])
        z_axis = np.cross(x_axis, mouth_middle - l_eye_middle)
        y_axis = np.cross(z_axis, x_axis)
        z_axis = normalize(z_axis)
        y_axis = normalize(y_axis)
        forth_matrix = np.array([x_axis, y_axis, z_axis]).transpose()
        back_matrix = np.linalg.inv(forth_matrix)
        return back_matrix

    def draw_eye_vector(self, drawer, eye_3d_vector, solver, rotation, translation, eye_pos):
        #eye_3d_vector[1] -= 0.1
        if eye_pos == "l":
            eye_pos = (solver.model_points_68[36] + solver.model_points_68[39]) / 2.
        else:
            eye_pos = (solver.model_points_68[42] + solver.model_points_68[45]) / 2.
       # eye_pos += [0., 0., -9.]
        eye_3d_vector = copy.deepcopy(eye_3d_vector) * -50
       # eye_3d_vector = np.matmul(back_matrix, eye_3d_vector)
        #if eye_3d_vector[2] < 0:
        #    eye_3d_vector *= -1
        no_rot_vector, _ = cv2.Rodrigues(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
        vect_start, _ = cv2.projectPoints(eye_pos, no_rot_vector, translation, solver.camera_matrix, solver.dist_coeefs)
        vect_finish, _ = cv2.projectPoints(eye_3d_vector + eye_pos, no_rot_vector, translation, solver.camera_matrix,
                                        solver.dist_coeefs)
        vect_start = vect_start.reshape((2,))
        vect_finish = vect_finish.reshape((2,))
        vect_finish = vect_start - vect_finish
        vect_start, _ = cv2.projectPoints(eye_pos, rotation, translation, solver.camera_matrix, solver.dist_coeefs)
        vect_start = vect_start.reshape((2,))
        drawer.line([*vect_start, vect_start[0] + vect_finish[0], vect_start[1] + vect_finish[1]], fill=(0, 255, 0))
        drawer.ellipse([vect_start[0] - 2, vect_start[1] - 2, vect_start[0] + 2, vect_start[1] + 2])
        """
        eye_3d_vector = copy.deepcopy(eye_3d_vector) * 50
        eye_3d_vector = np.matmul(back_matrix, eye_3d_vector)

        eye_vector_left = camera_holders.project_point(eye_3d_vector, solver, rotation, translation,
                                                       is_vector_translation=True)
        points = np.array([sum(np_points[:, 0]), sum(np_points[:, 1]), sum(np_points[:, 0]) , sum(np_points[:, 1]), ])
        points /= np_points.shape[0]
        points += [0, 0, eye_vector_left[0], + eye_vector_left[1]]
        points = list(map(lambda x: int(x), points))
        drawer.line(points, fill=(0, 255, 0))
"""