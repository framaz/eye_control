import json
import subprocess
import threading

import cv2
import numpy as np
import zerorpc

import drawer
from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from utilities import path_change_decorator
from .basic_predictor import BasicPredictor


def thread_func(debug_predictor, backend_server):
    while True:
        line = backend_server.stdout.readline()
        debug_predictor.kek = line
        try:
            json_dict = json.loads(line)
            debug_predictor.kek = json_dict
            if json_dict["type"] == "new_corner":
                drawer.button_callback()
            if json_dict["type"] == "gaze_vector_change":
                x = json_dict["value"]["x_right"]
                z = json_dict["value"]["z_right"]
                debug_predictor.right_eye_gaze = np.array([x, 1., z])
                x = json_dict["value"]["x_left"]
                z = json_dict["value"]["z_left"]
                debug_predictor.left_eye_gaze = np.array([x, 1., z])
            if json_dict["type"] == "matrix_change":
                matrix = json_dict["value"]
                matrix = np.array(list((map(lambda x: float(x), matrix[1: -1].split(', '))))).reshape((3, 4))
                debug_predictor.world_to_camera = matrix
                face_points = []
                for point in debug_predictor.non_transformed:
                    face_points.append(np.matmul(matrix, point))
                debug_predictor.face_points = np.array(face_points)
                debug_predictor.rotation_vector, _ = cv2.Rodrigues(matrix[:, :3])
        except json.decoder.JSONDecodeError:
            pass
        except TypeError:
            k = 3


class VisualDebugPredictor(BasicPredictor):
    @path_change_decorator
    def __init__(self, noised=True):
        self.backend = subprocess.Popen(["python", "backend.py"], stdout=subprocess.PIPE)
        frontend = subprocess.Popen(["./frontend/node_modules/.bin/electron", "./frontend"])
        self.noised = noised
        self.kek = "azaz"
        self.left_eye_gaze = np.array([0, -1, 0])
        self.right_eye_gaze = np.array([0, -1, 0])
        self.plane = np.array([0., 1., 0., -100.])
        thread = threading.Thread(target=thread_func, args=(self, self.backend))
        thread.start()
        self.solver = pnp_solver.PoseEstimator((720, 1280))
        self.client = zerorpc.Client()
        self.client.connect("tcp://127.0.0.1:4242")
        self.world_to_camera = np.zeros((3, 4), dtype=np.float64)
        self.world_to_camera[0, 0] = 1
        self.world_to_camera[1, 1] = 1
        self.world_to_camera[2, 2] = 1
        self.non_transformed = np.ones((68, 4))
        self.non_transformed[:, :-1] = np.copy(self.solver.model_points_68)
        self.non_transformed.transpose()
        self.face_points = self.solver.model_points_68
        self.rotation_vector = np.array([0., 0., 0.])

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        projections, _ = cv2.projectPoints(self.face_points, np.array([0., 0., 0.]), np.array([0., 0., 0.], ),
                                           self.solver.camera_matrix, self.solver.dist_coeefs)
        projections = projections.reshape((-1, 2))
        right_eye_gaze = np.copy(self.right_eye_gaze)
        left_eye_gaze = np.copy(self.left_eye_gaze)
        if self.noised:
            right_eye_gaze += np.random.normal(0, 0.1, (3,))
            left_eye_gaze += np.random.normal(0, 0.1, (3,))
            projections += np.random.normal(0, 2, (68, 2))
        return imgs, [right_eye_gaze], [left_eye_gaze], [projections], {}

    def get_mouse_coords(self, cameras, time_now):
        face, results, out_inform = self.predict(cameras, time_now)
        # out_inform["rotator"] -= angle
        # out_inform["cutter"] -= offset
        print(str(out_inform))

        result = list(results)
        results = [0, 0]
        for cam in result:
            for eye in cam:
                results[0] += eye[0]
                results[1] += eye[1]
        results[0] /= len(result) * 2
        results[1] /= len(result) * 2
        self.client.set_mouse_position(*results)