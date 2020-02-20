import collections

import matplotlib.pyplot as plt
import numpy as np

import camera_holders
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import utilities
from camera_holders.seen_to_screen import SeenToScreenTranslator
from utilities import smooth_n_cut, smooth_func

corner_dict = {"TL": 0, "TR": 1, "BR": 2, "BL": 3, "MM": 4}


class CameraSystemFactory:
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
            head_rotation_vect, head_translation_vect = self.solver.solve_pose(np_points)
            head_rotation_vect = np.append(head_rotation_vect, time_now)
            head_translation_vect = np.append(head_translation_vect, time_now)
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
        left = smooth_func(self.calibration_history_left, self.last_time)
        right = smooth_func(self.calibration_history_right, self.last_time)
        self.left_eye.corner_vectors[corner] = left
        self.right_eye.corner_vectors[corner] = right
        self.calibration_history_left = collections.deque()
        self.calibration_history_right = collections.deque()

    def calibration_final(self):
        rotation = smooth_n_cut(self.calibration_history_head_rotation, self.calibration_history_head_rotation[0][-1])
        translation = smooth_n_cut(self.calibration_history_head_translation,
                                   self.calibration_history_head_rotation[0][-1])
        head = camera_holders.Head(rotation, translation, self.solver)
        screen = camera_holders.Screen(self.left_eye, self.right_eye, self.solver, rotation, translation)
        world_to_camera = utilities.vector_to_camera_coordinate_system(rotation, translation)
        left_eye = sum(self.solver.model_points_68[36:41]) / 6
        right_eye = sum(self.solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)
        self.left_eye.create_translator(screen, right_eye)
        self.right_eye.create_translator(screen, left_eye)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(*left_eye, color="#ff0000")
        ax.scatter(*right_eye, color="#00ff00")
        kek = []
        for point in self.solver.model_points_68:
            kek.append(np.matmul(world_to_camera, [*point, 1]))
        kek = np.array(kek)
        kek = kek.transpose()
        ax.scatter(*kek, color="#0000ff")
        for i in range(5):
            draw_eyes(ax, left_eye, right_eye, self.left_eye.corner_vectors[i], self.right_eye.corner_vectors[i])
        # plt.show()
        return self.left_eye, self.right_eye, head


def draw_eyes(ax, left_eye, right_eye, left_vect, right_vect):
    ax.scatter(*(left_eye + 1000 * left_vect), color="#880000")
    ax.scatter(*(right_eye + 1000 * right_vect), color="#008800")


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
