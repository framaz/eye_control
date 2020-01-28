import json
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import subprocess
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import threading
import zerorpc
import numpy as np
import drawer


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


class DebugPredictor:
    def __init__(self):
        self.backend = subprocess.Popen(["python", "backend.py"], stdout=subprocess.PIPE)
        frontend = subprocess.Popen(["./frontend/node_modules/.bin/electron", "./frontend"])
        self.kek = "azaz"
        self.left_eye_gaze = np.array([0, -1, 0])
        self.right_eye_gaze = np.array([0, -1, 0])
        self.plane = np.array([0., 1., 0., -100.])
        thread = threading.Thread(target=thread_func, args=(self, self.backend))
        thread.start()
        self.client = zerorpc.Client()
        self.client.connect("tcp://127.0.0.1:4242")

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        return [imgs], [self.right_eye_gaze], [self.left_eye_gaze], [np.zeros((68, 3))], {}


if __name__ == "__main__":
    pred = DebugPredictor()
    while True:
        print(pred.gaze_vector)
        pass
