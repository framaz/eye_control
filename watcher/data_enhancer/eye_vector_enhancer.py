from __future__ import annotations

import copy

import cv2
import numpy as np
import typing
from PIL import ImageDraw, Image

from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from utilities import get_world_to_camera_matrix
from .landmarks_avg_person_enhancer import LandmarksAvgPersonEnhancer


def draw_eye_vector(drawer: ImageDraw.ImageDraw, eye_3d_vector: np.ndarray,
                    solver: pnp_solver.PoseEstimator,
                    rotation: np.ndarray, translation: np.ndarray, eye_pos: str) -> None:
    """Draw gaze vector eye from eye on a picture drawer

    :param drawer: where to draw
    :param eye_3d_vector: what vector(in camera coordinate system)
    :param solver:
    :param rotation: head rotation
    :param translation: head translation
    :param eye_pos: "l" for left, "r" for right
    :return:
    """
    if eye_pos == "l":
        eye_pos = (solver.model_points_68[36] + solver.model_points_68[39]) / 2.
    else:
        eye_pos = (solver.model_points_68[42] + solver.model_points_68[45]) / 2.

    eye_3d_vector = copy.deepcopy(eye_3d_vector) * -50

    # eye coordinate in camera coordinate system is determined
    to_camera_matrix = get_world_to_camera_matrix(solver, rotation, translation)
    eye_pos = np.matmul(to_camera_matrix, [*eye_pos, 1])

    no_rot_vector, _ = cv2.Rodrigues(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
    vect_start, _ = cv2.projectPoints(eye_pos, no_rot_vector, [0, 0, 0], solver.camera_matrix,
                                      solver.dist_coeefs)
    vect_finish, _ = cv2.projectPoints(eye_3d_vector + eye_pos, no_rot_vector, [0, 0, 0],
                                       solver.camera_matrix, solver.dist_coeefs)

    drawer.line([*vect_start, vect_start[0] + vect_finish[0], vect_start[1] + vect_finish[1]], fill=(0, 255, 0))
    drawer.ellipse([vect_start[0] - 2, vect_start[1] - 2, vect_start[0] + 2, vect_start[1] + 2])


class EyeVectorEnhancer(LandmarksAvgPersonEnhancer):
    """Draws eye gaze vectors"""

    def __init__(self, **kwargs):
        """Construct"""
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray,
                eye_vector_left: np.ndarray = None,
                eye_vector_right: np.ndarray = None) -> typing.Tuple[Image.Image, dict]:
        """Apply - draw eye gaze vectors

        :param pic:
        :param np_points: facial landmarks
        :param eye_vector_left: shape [3]
        :param eye_vector_right: shape [3]
        :return:
        """
        pic, output = super().process(pic, np_points, draw_points=True)

        solver = pnp_solver.PoseEstimator((720, 1080))

        drawer = ImageDraw.Draw(pic)

        rotation, translation = solver.solve_pose(np_points)

        draw_eye_vector(drawer, eye_vector_left, solver, rotation, translation, eye_pos="l")
        draw_eye_vector(drawer, eye_vector_right, solver, rotation, translation, eye_pos="r")
        return pic, output
