import cv2
import numpy as np
from PIL import ImageDraw, Image

import utilities
from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from .data_enhancer import DataEnhancer


class HeadAxisEnhancer(DataEnhancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, pic, np_points, draw_points=False):
        pic, output = super().process(pic, np_points)
        solver = pnp_solver.PoseEstimator((720, 1280))
        rotation, translation = solver.solve_pose(np_points)
        if draw_points:
            drawer = ImageDraw.Draw(pic)
            solver = pnp_solver.PoseEstimator((720, 1280))
            np_points = np_points.astype(dtype=np.float64)
            proj_points, _ = cv2.projectPoints(solver.model_points_68, rotation, translation, solver.camera_matrix, solver.dist_coeefs)
            proj_points = proj_points.astype(dtype=np.int64)
            for point in proj_points:
                [res] = point
                #drawer.ellipse([res[0] - 2, res[1] - 2, res[0] + 2, res[1] + 2])
            matrix = utilities.get_world_to_camera_matrix(solver, rotation, translation, )
            for point in solver.model_points_68:
                new_point = np.matmul(matrix, [*point, 1])
                new_point /= new_point[2]
                res = new_point
                drawer.ellipse([res[0] - 2, res[1] - 2, res[0] + 2, res[1] + 2])
        pic = solver.draw_axes(pic, rotation, translation)
        pic = pic.astype(np.uint8)
        pic = Image.fromarray(pic)
        return pic, output