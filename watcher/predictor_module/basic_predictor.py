from __future__ import annotations

import pyautogui
import typing
from PIL import Image
import numpy as np


class BasicPredictor:
    """A class of basic predictors

    It doesnt implement gaze vector prediction at all. All subclasses should implement it.
    """

    def predict_eye_vector_and_face_points(
            self,
            imgs: typing.List[Image.Image],
            time_now: float) -> typing.Tuple[
                typing.List[Image.Image],
                typing.List[np.ndarray],
                typing.List[np.ndarray],
                typing.List[np.ndarray],
                dict]:
        """Not supposed to be called from basic calss

        :param imgs: list of images to predict from
        :param time_now: current time
        :return: tuple of shape [5], where:
                    [0] - list of resulting images
                    [1] - list of right eye gazes of shape [3]
                    [2] - list of left eye gazes of shape [3]
                    [3] - list of facial landmark points [68,2]
                    [4] - dict of some output information
        """
        raise NotImplementedError

    def move_mouse_to_gaze_pixel(self, cameras: typing.List[CameraHolder], time_now: float) -> None:
        """Find a target gaze pixel for all cameras and move cursor there

        If gaze is out of bounds then it is fit to the bounds

        :param cameras: list of all CameraHolders
        :param time_now: current time (time.time())
        :return:
        """
        face, results, out_inform = self.predict(cameras, time_now)

        result = list(results)
        results = [0, 0]
        for cam in result:
            for eye in cam:
                results[0] += eye[0]
                results[1] += eye[1]

        results[0] /= len(result) * 2
        results[1] /= len(result) * 2

        if results[0] < 0:
            results[0] = 0
        if results[1] < 0:
            results[1] = 0
        if results[0] >= 1920:
            results[0] = 1919
        if results[1] >= 1080:
            results[1] = 1079

        pyautogui.moveTo(results[0], results[1])

    def predict(self, cameras: typing.List[CameraHolder],
                time_now: float) -> (typing.List[Image.Image], typing.List[np.ndarray], dict):
        """For all cameras predict their pixel gaze targets

        :param cameras: list of all cameras
        :param time_now: current time (time.time())
        :return: a tuple of shape [3], where:
                    [0] - list of face pictures
                    [1] - list of pixel targets
                    [2] - dict of out inform
        """
        imgs = []
        for camera in cameras:
            imgs.append(camera.get_picture())

        faces, eye_one_vectors, eye_two_vectors, np_points_all, tmp_out_informs = \
            self.predict_eye_vector_and_face_points(imgs, time_now, )

        results = []
        for i, camera in zip(range(len(cameras)), cameras):
            results.append(camera.get_screen_positions(eye_one_vectors[i],
                                                       eye_two_vectors[i],
                                                       np_points_all[i],
                                                       time_now))

        def face_to_img(face):
            """Convert picture to PIL.Image.Image"""
            if not isinstance(face, Image.Image):
                face = Image.fromarray(face)
            return face

        faces = list(map(lambda face: face_to_img(face), faces))
        return faces, results, tmp_out_informs
