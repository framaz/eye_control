import functools
import json
import subprocess
import threading
import time

import numpy as np
import typing
from PIL import Image

from .basic_predictor import BasicPredictor


class GazeMLPredictor(BasicPredictor):
    """GazeML based predictor

    Actually, it is just an interface for gazeML subprocess as i was too lazy to implement
    in a proper way.
    All the information is got from gazeml subprocess.

    :ivar gazeML_process: (subprocess.Popen), running gazeml subprocess
    :ivar eye_one_target: (np.ndarray, shape [3]), gaze vector from first eye
    :ivar eye_two_target: (np.ndarray, shape [3]), gaze vector from second eye
    :ivar np_points: (np.ndarray, shape [68, 2]), current facial landmarks
    """
    def __init__(self):
        """Constructor

        Runs the subprocess and a thread for reading the data from subprocess
        """
        self.gazeML_process = subprocess.Popen(
            ["python", "./from_internet_or_for_from_internet/GazeML/src/elg_demo.py"],
            stdout=subprocess.PIPE)

        self.eye_one_target = np.array([0, 0, 1])
        self.eye_two_target = np.array([0, 0, 1])
        self.np_points = np.zeros((68, 2))

        def read_from_subprocess(predictor: GazeMLPredictor, process: subprocess.Popen):
            """Read info from subprocess stdout and change the predictor values according to it

            :param predictor: predictor to write
            :param process: process where to read from
            :return:
            """
            while True:
                line = process.stdout.readline()
                try:
                    data = json.loads(line)

                    # eye number 0 is second eye
                    # it was observed that y should be inverted
                    if data["eye_number"] == 0:
                        predictor.eye_two_target = np.array([data["x"], -data["y"], data["z"]])
                    else:
                        predictor.eye_one_target = np.array([data["x"], -data["y"], data["z"]])

                    predictor.np_points = np.array(data['np_points']).reshape((68, 2))
                except json.decoder.JSONDecodeError:
                    pass

        thread_func = functools.partial(read_from_subprocess, self, self.gazeML_process)
        thread = threading.Thread(target=thread_func)
        thread.start()

    def predict_eye_vector_and_face_points(
            self,
            imgs: typing.List[Image.Image],
            time_now: int) -> typing.Tuple[
                typing.List[Image.Image],
                typing.List[np.ndarray],
                typing.List[np.ndarray],
                typing.List[np.ndarray],
                dict]:
        """Retrieves the gaze vectors and facial landmarks

        imgs and time_now aren't really needed, they are just for LSP

        :param imgs: list of images to predict from
        :param time_now: current time
        :return: tuple of shape [5], where:
                    [0] - list of resulting images
                    [1] - list of right eye gazes of shape [3]
                    [2] - list of left eye gazes of shape [3]
                    [3] - list of facial landmark points [68,2]
                    [4] - dict of some output information
        """
        time.sleep(0.2)
        return [Image.fromarray(np.zeros((720, 1280)))], [self.eye_one_target],\
               [self.eye_two_target], [self.np_points], {}
