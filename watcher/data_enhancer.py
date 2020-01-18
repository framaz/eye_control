import copy
import time
from math import sqrt

import numpy as np
import camera_holders
from PIL import Image, ImageDraw, ImageFont
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
from predictor_module import normalize
import cv2

class AbsDataEnhancer:
    def __init__(self, **kwargs):
        pass

    def process(self, pic, np_points):
        if type(self) == AbsDataEnhancer:
            raise NotImplementedError("AbsDataEnhancer process call")

        if not isinstance(pic, Image.Image):
            pic = pic.astype(dtype=np.uint8)
            pic = Image.fromarray(pic)
        pic = pic.convert('RGB')
        output = {}
        return pic, output


class PointDataEnhancer(AbsDataEnhancer):
    def __init__(self, point_size=2, **kwargs):
        super().__init__(**kwargs)
        self.point_radius = point_size

    def process(self, pic, np_points):
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        for [x, y] in np_points:
            drawer.ellipse([x - self.point_radius, y - self.point_radius, x + self.point_radius, y + self.point_radius],
                           fill=0)
        return pic, output


class WidthHeightDataEnhancer(PointDataEnhancer):
    def __init__(self, line_width=2, text_size=9, **kwargs):
        self.text_size = text_size
        self.line_width = line_width
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray):
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        font = ImageFont.truetype("arial.ttf", size=self.text_size)
        for first, second in head_tracker.width_metrics + head_tracker.heigth_metrics:
            first = np_points[first]
            second = np_points[second]
            drawer.line([*first, *second], width=self.line_width)
            [text_x, text_y] = second
            text_x += 2
            val = sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)
            drawer.text([text_x, text_y], str(int(val)), font=font, fill=(255, 0, 0))
        return pic, output


if __name__ == "__main__":
    lel = WidthHeightDataEnhancer(kek=20)
    ImageFont.truetype("arial.ttf", size=8)
    lel = lel


class LeftTopCornerTextWriteDataEnhancer(AbsDataEnhancer):
    def __init__(self, text_size=9, **kwargs):
        self.text_size = text_size
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray, obj_to_out="kek"):
        text = str(obj_to_out)
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        font = ImageFont.truetype("arial.ttf", size=self.text_size)
        drawer.multiline_text([0, 0], text, fill=(255, 0, 0), font=font)
        return pic, output


class HeadPositionAxisDataEnhancer(AbsDataEnhancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, pic, np_points, draw_points=False):
        pic, output = super().process(pic, np_points)
        solver = pnp_solver.PoseEstimator((1080, 1920))
        rotation, translation = solver.solve_pose(np_points)
        if draw_points:
            drawer = ImageDraw.Draw(pic)
            solver = pnp_solver.PoseEstimator((1080, 1920))
            np_points = np_points.astype(dtype=np.float64)
            proj_points, _ = cv2.projectPoints(solver.model_points_68, rotation, translation, solver.camera_matrix, solver.dist_coeefs)
            proj_points = proj_points.astype(dtype=np.int64)
            for point in proj_points:
                [res] = point
                #drawer.ellipse([res[0] - 2, res[1] - 2, res[0] + 2, res[1] + 2])
            matrix = camera_holders.get_world_to_camera_matrix(solver, rotation, translation, )
            for point in solver.model_points_68:
                new_point = np.matmul(matrix, [*point, 1])
                new_point /= new_point[2]
                res = new_point
                drawer.ellipse([res[0] - 2, res[1] - 2, res[0] + 2, res[1] + 2])
        pic = solver.draw_axes(pic, rotation, translation)
        pic = pic.astype(np.uint8)
        pic = Image.fromarray(pic)
        return pic, output


class HeadNEyeDataEnhancer(HeadPositionAxisDataEnhancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, pic, np_points, eye_vector_left, eye_vector_right):
        pic, output = super().process(pic, np_points, draw_points=False)
        solver = pnp_solver.PoseEstimator((1080, 1920))
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
        eye_3d_vector[1] -= 0.1
        if eye_pos == "l":
            eye_pos = (solver.model_points_68[36] + solver.model_points_68[39]) / 2.
        else:
            eye_pos = (solver.model_points_68[42] + solver.model_points_68[45]) / 2.
        eye_pos += [0., 0., -9.]
        eye_3d_vector = copy.deepcopy(eye_3d_vector) * 50
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

class EyeVectorSurfaceProjectorDataEnhancer(AbsDataEnhancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, pic, np_points, eye_vector_left, eye_vector_right):
        pic, output = super().process(pic, np_points)
        solver = pnp_solver.PoseEstimator((1080, 1920))
        drawer = ImageDraw.Draw(pic)
        drawer.rectangle([0, 0, 1920, 1080], fill=(255, 255, 255))
        middle = [960, 540]
        eye_vector_right *= 1000
        eye_vector_left *= 1000
        drawer.line([middle[0], middle[1], middle[0] + eye_vector_left[0], middle[1] + eye_vector_left[2]], fill=(0,255,0))
        drawer.line([middle[0], middle[1], middle[0] + eye_vector_right[0], middle[1] + eye_vector_right[2]], fill=(0,255,0))
        return pic, output