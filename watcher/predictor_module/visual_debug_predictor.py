import json
import subprocess
import threading

import cv2
import numpy as np
import typing
import zerorpc
from PIL import Image

import drawer
from camera_holders import CameraHolder
from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from utilities import path_change_decorator
from .basic_predictor import BasicPredictor


class VisualDebugPredictor(BasicPredictor):
    """A watcher representation of visual debug predictor

    :ivar _backend: (subprocess.Popen), backend for debug predictor
    :ivar _noised: (bool), whether noising of gaze vectors is needed
    :ivar _left_eye_gaze: (np.ndarray, shape [3]), left eye gaze vector
    :ivar _right_eye_gaze: (np.ndarray, shape [3]), right eye gaze vector
    :ivar _plane: (np.ndarray, shape [4]), plane ax + by + cz + d = 0 in form [a, b, c, d]
    :ivar _solver: (pnp_solver.PoseEstimator)
    :ivar _client: (zerorpc.Client): zerorpc client to communicate with backend
    :ivar _world_to_camera: (np.ndarray, shape [3, 4]), face to camera translation matrix
    :ivar _non_transformed: (np.ndarray, shape [68, 3]), face landmarks in face coordinate system
    :ivar _face_points: (np.ndarray, shape [68, 3]), current face landmark pos in camera coordinate system
    :ivar
    :ivar

    """
    @path_change_decorator
    def __init__(self, noised: bool = True):
        """Construct object

        Constructor runs both backend and frontend subprocess
        Backend can't be run in a thread as gevent(zerorpc) doesn't like threads

        :param noised: whether to noise the predicted gaze vectors and fake landmarks
        """
        self._backend = subprocess.Popen(["python", "backend.py"], stdout=subprocess.PIPE)
        subprocess.Popen(["./frontend/node_modules/.bin/electron", "./frontend"])

        self._noised = noised

        self._left_eye_gaze = np.array([0, -1, 0])
        self._right_eye_gaze = np.array([0, -1, 0])
        self._plane = np.array([0., 1., 0., -100.])

        thread = threading.Thread(target=debug_stdout_listener, args=(self, self._backend))
        thread.start()
        self._client = zerorpc.Client()
        self._client.connect("tcp://127.0.0.1:4242")

        self._solver = pnp_solver.PoseEstimator((720, 1280))

        self._world_to_camera = np.zeros((3, 4), dtype=np.float64)
        self._world_to_camera[0, 0] = 1
        self._world_to_camera[1, 1] = 1
        self._world_to_camera[2, 2] = 1

        self._non_transformed = np.ones((68, 4))
        self._non_transformed[:, :-1] = np.copy(self._solver.model_points_68)
        self._non_transformed.transpose()

        self._face_points = self._solver.model_points_68

    def predict_eye_vector_and_face_points(
            self,
            imgs: typing.List[Image.Image],
            time_now: float) -> typing.Tuple[
                typing.List[Image.Image],
                typing.List[np.ndarray],
                typing.List[np.ndarray],
                typing.List[np.ndarray],
                dict]:
        """Get current gaze position and face 2d landmarks on pic

        Noise is applied here if _noised is set as true

        :param imgs: list of images
        :param time_now: current time
        :return:
        """
        projections, _ = cv2.projectPoints(self._face_points, np.array([0., 0., 0.]),
                                           np.array([0., 0., 0.]), self._solver.camera_matrix,
                                           self._solver.dist_coeefs)
        projections = projections.reshape((-1, 2))

        right_eye_gaze = np.copy(self._right_eye_gaze)
        left_eye_gaze = np.copy(self._left_eye_gaze)

        if self._noised:
            right_eye_gaze += np.random.normal(0, 0.1, (3,))
            left_eye_gaze += np.random.normal(0, 0.1, (3,))
            projections += np.random.normal(0, 2, (68, 2))

        return imgs, [right_eye_gaze], [left_eye_gaze], [projections], {}

    def move_mouse_to_gaze_pixel(self, cameras: typing.List[CameraHolder], time_now: float):
        """Get mouse position on on screen

        In this class it just notifies the backend about changing of pixel target

        :param cameras: list of cameras
        :param time_now: current time
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

        self._client.set_mouse_position(*results)


def debug_stdout_listener(debug_predictor: VisualDebugPredictor, backend_server: subprocess.Popen):
    """Listen to backend_server stdout and modify visualDebugPredictor

    Should be called as a thread function

    :param debug_predictor:
    :param backend_server:
    :return:
    """
    while True:
        line = backend_server.stdout.readline()
        try:
            json_dict = json.loads(line)

            # new corner signal
            if json_dict["type"] == "new_corner":
                drawer.button_callback()

            # gaze vector change signal
            if json_dict["type"] == "gaze_vector_change":
                x = json_dict["value"]["x_right"]
                z = json_dict["value"]["z_right"]
                debug_predictor._right_eye_gaze = np.array([x, 1., z])

                x = json_dict["value"]["x_left"]
                z = json_dict["value"]["z_left"]
                debug_predictor._left_eye_gaze = np.array([x, 1., z])

            # face position change
            if json_dict["type"] == "matrix_change":
                matrix = json_dict["value"]
                matrix = np.array(list((map(lambda x: float(x), matrix[1: -1].split(', '))))).reshape((3, 4))

                debug_predictor._world_to_camera = matrix

                face_points = []
                for point in debug_predictor._non_transformed:
                    face_points.append(np.matmul(matrix, point))

                debug_predictor._face_points = np.array(face_points)

        except json.decoder.JSONDecodeError:
            pass
        except TypeError:
            pass
