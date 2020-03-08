import collections

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import typing

import camera_holders
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import utilities
from .eye import Eye
from .head import Head
from .seen_to_screen import SeenToScreenTranslator
from predictor_module import BasicPredictor
from utilities import smooth_n_cut, smooth_func

corner_dict = {"TL": 0, "TR": 1, "BR": 2, "BL": 3, "MM": 4}


class CameraSystemFactory:
    """Accumulate all parameters(head pos, gazes) tick by tick, construct camera system

    For better results person shouldn't move head for the whole process of camera system creation.
    Then person should concentrate his gaze on one of the corners of the screen(order is in corner_dict)
    To create a camera system one should call calibrate_remember to remember current
    gaze vector and head position many times till the values stop changing. When they are stopped
    changing, one should call calibration_end to summarise the corner.
    That process should be repeated for each corner in  corner_dict.
    After the end calibration_final is called to end finally create all the eyes, screen, head etc.

    :ivar _solver: (pnp_solver.PoseEstimator) used to get 68 face points in face coord system
    :ivar _calibration_history_left: (collections.deque) format [x, y, z, time]
                                     used to store history of left eye gaze
    :ivar _calibration_history_right: (collections.deque) format [x, y, z, time]
                                      used to store history of right eye gaze
    :ivar _calibration_history_head_translation: (collections.deque) format [tx, ty, tz, time]
                                              used to store history of head translation
    :ivar _calibration_history_head_rotation: (collections.deque) format [rx, ry, rz, time]
                                              used to store history of head rotation(rotation vector)
    :ivar _last_time: when last calibration tick was
    """
    def __init__(self, solver: pnp_solver.PoseEstimator):
        """Constructor

        :param solver:
        """
        self._solver = solver

        self._calibration_history_left = collections.deque()
        self._calibration_history_right = collections.deque()
        self._calibration_history_head_rotation = collections.deque()
        self._calibration_history_head_translation = collections.deque()

        self._last_time = 0

        self._left_eye = camera_holders.Eye()
        self._right_eye = camera_holders.Eye()

    def calibrate_remember(self,
                           img: PIL.Image.Image,
                           time_now: float,
                           predictor: BasicPredictor) -> str:
        """Makes a tick of remembering of current corner gaze

        :param img: pic of a person taken by camera in original scale
        :param time_now: current time (time.time())
        :param predictor: predictor for current camera system, used only to get gaze and
                          face_points
        :return:
        """
        result = predictor.predict_eye_vector_and_face_points(img, time_now)
        self._last_time = time_now

        if not (result is None):
            _, [left_eye_vect], [right_eye_vect], [np_points], _ = result

            head_rotation_vect, head_translation_vect = self._solver.solve_pose(np_points)
            head_rotation_vect = np.append(head_rotation_vect, time_now)
            head_translation_vect = np.append(head_translation_vect, time_now)
            left_eye_vect = np.append(left_eye_vect, time_now)
            right_eye_vect = np.append(right_eye_vect, time_now)

            self._calibration_history_left.append(left_eye_vect)
            self._calibration_history_right.append(right_eye_vect)
            self._calibration_history_head_rotation.append(head_rotation_vect)
            self._calibration_history_head_translation.append(head_translation_vect)

        return str(smooth_n_cut(self._calibration_history_left, time_now)) + \
               str(smooth_n_cut(self._calibration_history_right, time_now)) + \
               str(smooth_n_cut(self._calibration_history_head_translation, time_now)) + \
               str(smooth_n_cut(self._calibration_history_head_rotation, time_now))

    def calibration_end(self, corner: typing.Union[int, str]) -> None:
        """Complete current corner gaze vector remembering

        :param corner: corner number or string
        :return:
        """
        if isinstance(corner, str):
            corner = corner_dict[corner]

        left = smooth_func(self._calibration_history_left, self._last_time)
        right = smooth_func(self._calibration_history_right, self._last_time)

        self._left_eye.corner_vectors[corner] = left
        self._right_eye.corner_vectors[corner] = right

        self._calibration_history_left = collections.deque()
        self._calibration_history_right = collections.deque()

    def calibration_final(self) -> typing.Tuple[Eye, Eye, Head]:
        """Finalise the creation of camera system.

        Creates Head, Screen, attaches screen to eyes and creates eye translators

        :return:
        """
        rotation = smooth_n_cut(self._calibration_history_head_rotation, self._calibration_history_head_rotation[0][-1])
        translation = smooth_n_cut(self._calibration_history_head_translation,
                                   self._calibration_history_head_rotation[0][-1])
        head = camera_holders.Head(self._solver)

        screen = camera_holders.Screen(self._left_eye, self._right_eye, self._solver, rotation, translation)

        # Eye centers are calculated to make translators
        world_to_camera = utilities.get_world_to_camera_matrix(rotation, translation)
        left_eye = sum(self._solver.model_points_68[36:41]) / 6
        right_eye = sum(self._solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)

        self._left_eye.create_translator(screen, right_eye)
        self._right_eye.create_translator(screen, left_eye)

        return self._left_eye, self._right_eye, head

# def create_translator(self):
#     self.left_eye.create_translator()
#     self.right_eye.create_translator()
#     return 0#SeenToScreenTranslator(self.points)


if __name__ == "__main__":
    trans = SeenToScreenTranslator([[0, 0], [100, 20], [120, -80], [-20, -100], [50, -50]])
    kek = trans.seen_to_screen([60, -40])
    azaza = trans.seen_to_screen([1, 1])
    azaza = azaza
