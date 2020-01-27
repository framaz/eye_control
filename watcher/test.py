import functools
import numpy as np
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import os
import threading
import zerorpc
import subprocess


class DebugState:
    pass

class DebugPredictor:
    def __init__(self):
        backend = subprocess.Popen(["python", "backend.py"], stdout=subprocess.PIPE)
        frontend = subprocess.Popen(["./frontend/node_modules/.bin/electron", "./frontend"])
        while True:
            print(backend.stdout.readline())

    def predict_eye_vector_and_face_points(self, imgs, time_now, configurator=None):
        return [imgs], [np.zeros((3,))], [np.zeros((3,))], [np.zeros((68, 2))], {}

if __name__ == "__main__":
    DebugPredictor()


