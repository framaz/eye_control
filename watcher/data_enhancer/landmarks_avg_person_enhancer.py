import cv2
import numpy as np
import typing
from PIL import ImageDraw, Image

import utilities
from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from .data_enhancer import DataEnhancer


class LandmarksAvgPersonEnhancer(DataEnhancer):
    """Draws projections of avg persons face on given rotation and translation

    Also draws axis"""

    def __init__(self, **kwargs):
        """Constructor"""
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray) -> typing.Tuple[Image.Image, dict]:
        """Apply enhancer - draw points and axis

        :param pic:
        :param np_points: facial landmarks
        :return:
        """
        pic, output = super().process(pic, np_points)
        solver = pnp_solver.PoseEstimator((720, 1280))
        rotation, translation = solver.solve_pose(np_points)

        drawer = ImageDraw.Draw(pic)

        proj_points, _ = cv2.projectPoints(solver.model_points_68, rotation, translation,
                                           solver.camera_matrix, solver.dist_coeefs)
        proj_points = proj_points.astype(dtype=np.int64)
        for point in proj_points:
            [res] = point
            drawer.ellipse([res[0] - 2, res[1] - 2, res[0] + 2, res[1] + 2])

        pic = solver.draw_axes(pic, rotation, translation)
        pic = pic.astype(np.uint8)
        pic = Image.fromarray(pic)
        return pic, output
