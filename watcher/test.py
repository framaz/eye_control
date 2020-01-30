import json
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import subprocess
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import threading
import zerorpc
import numpy as np
import drawer
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import cv2
import predictor_module
import zerorpc

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
        except json.decoder.JSONDecodeError:
            pass
        except TypeError:
            k = 3


class DebugPredictor(predictor_module.BasicPredictor):
    def __init__(self):
        self.backend = subprocess.Popen(["python", "backend.py"], stdout=subprocess.PIPE)
        frontend = subprocess.Popen(["./frontend/node_modules/.bin/electron", "./frontend"])
        self.kek = "azaz"
        self.left_eye_gaze = np.array([0, -1, 0])
        self.right_eye_gaze = np.array([0, -1, 0])
        self.plane = np.array([0., 1., 0., -100.])
        thread = threading.Thread(target=thread_func, args=(self, self.backend))
        thread.start()
        self.solver = pnp_solver.PoseEstimator((1080, 1920))
        self.client = zerorpc.Client()
        self.client.connect("tcp://127.0.0.1:4242")

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        projections, _ = cv2.projectPoints(self.solver.model_points_68, np.array([0., 0., 0.]), np.array([0., 0., 0.],),
                                        self.solver.camera_matrix, self.solver.dist_coeefs)
        projections = projections.reshape((-1, 2))
        return imgs, [self.right_eye_gaze], [self.left_eye_gaze], [projections], {}

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


if __name__ == "__main__":
    pred = DebugPredictor()
    while True:
        pass
