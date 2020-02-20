import functools
import json
import subprocess
import threading
import time

import numpy as np

from .basic_predictor import BasicPredictor


class GazeMLPredictor(BasicPredictor):
    def __init__(self):
        self.gazeML_process = subprocess.Popen(
            ["python", "./from_internet_or_for_from_internet/GazeML/src/elg_demo.py"],
            stdout=subprocess.PIPE)
        self.eye_one_target = np.array([0, 0, 1])
        self.eye_two_target = np.array([0, 0, 1])
        self.np_points = np.zeros((68, 2))
        def read_from_subprocess(object, process):
            while True:
                line = process.stdout.readline()
                try:
                    dict = json.loads(line)
                    if dict["eye_number"] == 0:
                        object.eye_two_target = np.array([dict["x"], -dict["y"], dict["z"]])
                    else:
                        object.eye_one_target = np.array([dict["x"], -dict["y"], dict["z"]])
                    object.np_points = np.array(dict['np_points']).reshape((68, 2))
                except json.decoder.JSONDecodeError:
                    pass

        thread_func = functools.partial(read_from_subprocess, self, self.gazeML_process)
        thread = threading.Thread(target=thread_func)
        thread.start()

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        time.sleep(0.2)
        return [np.zeros((720, 1280))], [self.eye_one_target], [self.eye_two_target], [self.np_points], {}