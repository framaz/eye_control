import collections
import copy
import math
import imgaug as iaa
import numpy as np
from PIL import Image
import camera_holders
import eye_module
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
from predictor_module import GoodPredictor
from seen_to_screen import SeenToScreenTranslator

FIRST_SMOOTH_TIME = 1
SECOND_SMOOTH_TIME = 3

corner_dict = {"TL": 0, "TR": 1, "BR": 2, "BL": 3, "MM": 4}


def rotate(img, points, angle):
    if isinstance(img, Image.Image):
        img = np.array(img)
    kp = []
    for i in points:
        kp.append(iaa.Keypoint(x=i[0], y=i[1]))
    kps = iaa.KeypointsOnImage(kp, shape=img.shape)
    aug = iaa.augmenters.Affine(rotate=angle)
    img, kps = aug(image=img, keypoints=kps)
    return img, kps


def smooth_n_cut(time_list, time_now):
    while abs(time_list[0][-1] - time_now) > SECOND_SMOOTH_TIME:
        time_list.popleft()
    return smooth_func(time_list, time_now, FIRST_SMOOTH_TIME, SECOND_SMOOTH_TIME)


def smooth_func(time_list, time_now, first_part_time=FIRST_SMOOTH_TIME, second_part_time=SECOND_SMOOTH_TIME):
    if not isinstance(time_list, np.ndarray):
        time_list = np.array(time_list)
    ave = np.zeros((len(time_list[0]) - 1), dtype=np.float32)
    total_weight = 0.
    for cur_time in time_list:
        time = cur_time[-1]
        value = cur_time[0:len(cur_time) - 1]
        weight = 0
        if abs(time_now - time) < first_part_time:
            weight = 1
        elif abs(time_now - time) < second_part_time:
            time -= first_part_time
            tang = math.tan(1 / (second_part_time - first_part_time))
            weight = 1. - (time) * tang
        else:
            break
        ave += value * weight
        total_weight += weight
    if total_weight == 0:
        total_weight = 1
    ave = ave / total_weight
    return ave


class Calibrator:
    def __init__(self, solver: pnp_solver.PoseEstimator):
        self.solver = solver
        self.points = np.zeros((len(corner_dict), 2), dtype=np.float32)
        self.calibration_history_left = collections.deque()
        self.calibration_history_right = collections.deque()
        self.calibration_history_head_rotation = collections.deque()
        self.calibration_history_head_translation = collections.deque()
        self.last_time = 0
        self.left_eye = camera_holders.Eye(eye_type='l')
        self.right_eye = camera_holders.Eye(eye_type='r')

    def calibrate_remember(self, img, time_now, predictor):
        result = predictor.predict_eye_vector_and_face_points(img, time_now)
        self.last_time = time_now
        if not (result is None):
            _, [left_eye_vect], [right_eye_vect], [np_points], _ = result
            left_eye_vect = np.append(left_eye_vect, time_now)
            right_eye_vect = np.append(right_eye_vect, time_now)
            if isinstance(predictor, GoodPredictor):
                head_rotation_vect, head_translation_vect = self.solver.solve_pose(np_points)
                head_rotation_vect = np.append(head_rotation_vect, time_now)
                head_translation_vect = np.append(head_translation_vect, time_now)
            else:
                head_rotation_vect = np.array([0., 0., 0., time_now])
                head_translation_vect = head_rotation_vect
            self.calibration_history_left.append(left_eye_vect)
            self.calibration_history_right.append(right_eye_vect)
            self.calibration_history_head_rotation.append(head_rotation_vect)
            self.calibration_history_head_translation.append(head_translation_vect)
        return str(smooth_n_cut(self.calibration_history_left, time_now)) + \
               str(smooth_n_cut(self.calibration_history_right, time_now)) + \
               str(smooth_n_cut(self.calibration_history_head_translation, time_now)) + \
               str(smooth_n_cut(self.calibration_history_head_rotation, time_now))

    def calibration_end(self, corner):
        if isinstance(corner, str):
            corner = corner_dict[corner]
        left = smooth_func(self.calibration_history_left, self.last_time, FIRST_SMOOTH_TIME,
                           SECOND_SMOOTH_TIME)
        right = smooth_func(self.calibration_history_right, self.last_time, FIRST_SMOOTH_TIME,
                            SECOND_SMOOTH_TIME)
        self.left_eye.corner_vectors[corner] = left
        self.right_eye.corner_vectors[corner] = right
        self.calibration_history_left = collections.deque()
        self.calibration_history_right = collections.deque()

    def calibration_final(self):
        rotation = smooth_n_cut(self.calibration_history_head_rotation, self.calibration_history_head_rotation[0][-1])
        translation = smooth_n_cut(self.calibration_history_head_translation, self.calibration_history_head_rotation[0][-1])
        head = camera_holders.Head(rotation, translation, self.solver)
        screen = camera_holders.Screen(self.left_eye, self.right_eye, self.solver, rotation, translation)
        world_to_camera = camera_holders.vector_to_camera_coordinate_system(rotation, translation)
        left_eye = sum(self.solver.model_points_68[36:41]) / 6
        right_eye = sum(self.solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)
        self.left_eye.create_translator(screen, right_eye)
        self.right_eye.create_translator(screen, left_eye)
        return self.left_eye, self.right_eye, head


# def create_translator(self):
#     self.left_eye.create_translator()
#     self.right_eye.create_translator()
#     return 0#SeenToScreenTranslator(self.points)


if __name__ == "__main__":
    trans = SeenToScreenTranslator([[0, 0], [100, 20], [120, -80], [-20, -100], [50, -50]])
    kek = trans.seen_to_screen([60, -40])
    azaza = trans.seen_to_screen([1, 1])
    azaza = azaza
PADDING = 10


class BasicTranslation:
    def __init__(self, np_points):
        self.np_points_before = copy.deepcopy(np_points)

    def apply_forth(self, face):
        raise NotImplementedError
        pass

    def apply_back(self):
        raise NotImplementedError
        pass

    def __str__(self):
        return type(self).__name__

    def get_modification_data(self):
        raise NotImplementedError


class FaceCropper(BasicTranslation):
    def __init__(self, np_points, face=None):
        # TODO save width/height of pic before cropping
        super().__init__(np_points)
        self.face = face
        self.left = min(np_points[:, 0]) - PADDING
        self.right = max(np_points[:, 0]) + PADDING
        self.top = min(np_points[:, 1]) - PADDING
        self.bottom = max(np_points[:, 1]) + PADDING

        offset = np.array([min(np_points[:, 0]) - PADDING, min(np_points[:, 1]) - PADDING],
                          dtype=np.float32)
        for i in range(np_points.shape[0]):
            np_points[i] -= offset
        self.offset = offset
        self.result_offset = None
        self.np_points_after = copy.deepcopy(np_points)
        self.width = 0
        self.heigth = 0
        self.resize_ratio_x = 0
        self.resize_ratio_y = 0

    def apply_forth(self, face):
        if not (isinstance(face, np.ndarray)):
            face = np.array(face)
        np_points = copy.deepcopy(self.np_points_before)
        if not (self.face is None):
            [self.width, self.heigth] = self.face.shape
            self.resize_ratio_x = face.shape[0] / self.width
            self.resize_ratio_y = face.shape[1] / self.heigth
            np_points *= [self.resize_ratio_x, self.resize_ratio_y]
        if type(face) is np.ndarray:
            face = Image.fromarray(face)
        face = face.crop((min(np_points[:, 0]) - PADDING, min(np_points[:, 1]) - PADDING,
                          max(np_points[:, 0]) + PADDING, max(np_points[:, 1]) + PADDING,))
        self.result_offset = min(np_points[:, 0]) - PADDING, min(np_points[:, 1]) - PADDING
        return face, copy.deepcopy(self.np_points_after)

    def __str__(self):
        return f"{super().__str__()}: {self.offset}"

    def get_modification_data(self):
        return self.offset,

    def get_resulting_offset(self):
        return self.result_offset


class RotationTranslation(BasicTranslation):
    def __init__(self, np_points):
        super().__init__(np_points)
        self.middle = None
        vect, self.middle = eye_module.find_eye_middle(np_points)
        self.angle = np.angle([complex(vect[0], vect[1])], deg=True)
        if self.angle > 90:
            self.angle -= 180
        offset = np.array([min(np_points[:, 0]) - PADDING, min(np_points[:, 1]) - PADDING],
                          dtype=np.float32)
        self.middle -= offset
        self.np_points_after = None

    def apply_forth(self, face):
        np_points = copy.deepcopy(self.np_points_before)
        face, kps = rotate(face, np_points, - self.angle)
        for i in range(len(kps.keypoints)):
            np_points[i, 0] = kps.keypoints[i].x
            np_points[i, 1] = kps.keypoints[i].y
        self.np_points_after = np_points
        return face, copy.deepcopy(np_points)

    def __str__(self):
        return f"{super().__str__()}: {self.angle}"

    def get_modification_data(self):
        return self.angle
