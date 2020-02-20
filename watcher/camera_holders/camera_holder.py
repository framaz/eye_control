import base64
import signal
import subprocess
from io import BytesIO

import PIL
import cv2
import gevent
import numpy as np
import zerorpc
from PIL import Image

import utilities
from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from utilities import vector_to_camera_coordinate_system
from .camera_system_factory import CameraSystemFactory


class CameraHolder:
    class CameraCalibrationServer:
        def __init__(self, camera):
            self.server = None
            self.camera = camera
            self.attributes = {}
            self.attribute_names = {
                "CAP_PROP_POS_MSEC": cv2.CAP_PROP_POS_MSEC,
                "CAP_PROP_POS_FRAMES": cv2.CAP_PROP_POS_FRAMES,
                "CAP_PROP_POS_AVI_RATIO": cv2.CAP_PROP_POS_AVI_RATIO,
                "CAP_PROP_FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
                "CAP_PROP_FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
                "CAP_PROP_FPS": cv2.CAP_PROP_FPS,
                "CAP_PROP_FOURCC": cv2.CAP_PROP_FOURCC,
                "CAP_PROP_FRAME_COUNT": cv2.CAP_PROP_FRAME_COUNT,
                "CAP_PROP_FORMAT": cv2.CAP_PROP_FORMAT,
                "CAP_PROP_MODE": cv2.CAP_PROP_MODE,
                "CAP_PROP_BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
                "CAP_PROP_CONTRAST": cv2.CAP_PROP_CONTRAST,
                "CAP_PROP_SATURATION": cv2.CAP_PROP_SATURATION,
                "CAP_PROP_HUE": cv2.CAP_PROP_HUE,
                "CAP_PROP_GAIN": cv2.CAP_PROP_GAIN,
                "CAP_PROP_EXPOSURE": cv2.CAP_PROP_EXPOSURE,
                "CAP_PROP_CONVERT_RGB": cv2.CAP_PROP_CONVERT_RGB,
                "CAP_PROP_WHITE_BALANCE_BLUE_U": cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
                "CAP_PROP_RECTIFICATION": cv2.CAP_PROP_RECTIFICATION,
                "CAP_PROP_MONOCHROME": cv2.CAP_PROP_MONOCHROME,
                "CAP_PROP_SHARPNESS": cv2.CAP_PROP_SHARPNESS,
                "CAP_PROP_AUTO_EXPOSURE": cv2.CAP_PROP_AUTO_EXPOSURE,
                "CAP_PROP_GAMMA": cv2.CAP_PROP_GAMMA,
                "CAP_PROP_TEMPERATURE": cv2.CAP_PROP_TEMPERATURE,
                "CAP_PROP_TRIGGER": cv2.CAP_PROP_TRIGGER,
                "CAP_PROP_TRIGGER_DELAY": cv2.CAP_PROP_TRIGGER_DELAY,
                "CAP_PROP_WHITE_BALANCE_RED_V": cv2.CAP_PROP_WHITE_BALANCE_RED_V,
                "CAP_PROP_ZOOM": cv2.CAP_PROP_ZOOM,
                "CAP_PROP_FOCUS": cv2.CAP_PROP_FOCUS,
                "CAP_PROP_GUID": cv2.CAP_PROP_GUID,
                "CAP_PROP_ISO_SPEED": cv2.CAP_PROP_ISO_SPEED,
                "CAP_PROP_BACKLIGHT": cv2.CAP_PROP_BACKLIGHT,
                "CAP_PROP_PAN": cv2.CAP_PROP_PAN,
                "CAP_PROP_TILT": cv2.CAP_PROP_TILT,
                "CAP_PROP_ROLL": cv2.CAP_PROP_ROLL,
                "CAP_PROP_IRIS": cv2.CAP_PROP_IRIS,
                "CAP_PROP_SETTINGS": cv2.CAP_PROP_SETTINGS,
                "CAP_PROP_BUFFERSIZE": cv2.CAP_PROP_BUFFERSIZE,
                "CAP_PROP_AUTOFOCUS": cv2.CAP_PROP_AUTOFOCUS,
                "CAP_PROP_SAR_NUM": cv2.CAP_PROP_SAR_NUM,
                "CAP_PROP_SAR_DEN": cv2.CAP_PROP_SAR_DEN,
                "CAP_PROP_BACKEND": cv2.CAP_PROP_BACKEND,
                "CAP_PROP_CHANNEL": cv2.CAP_PROP_CHANNEL,
                "CAP_PROP_AUTO_WB": cv2.CAP_PROP_AUTO_WB,
                "CAP_PROP_WB_TEMPERATURE": cv2.CAP_PROP_WB_TEMPERATURE,
            }
            for attr_name in self.attribute_names:
                i = self.attribute_names[attr_name]
                res = self.camera.get(i)
                if res != -1:
                    self.attributes[i] = res, attr_name

        def add_server(self, server):
            self.server = server
            self.server.debug = True
            gevent.signal(signal.SIGTERM, self.server.stop)

        def exit(self):
            self.server.stop()

        def get_attributes(self):
            return self.attributes

        def set_attribute(self, attribute, value):
            attribute = int(attribute)
            value = int(value)
            self.camera.set(attribute, value)

        def get_frame(self):
            ret, image = self.camera.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #           image = cv2.fastNlMeansDenoising(image.reshape((*(image.shape), 1)))
            buffered = BytesIO()
            image = Image.fromarray(image)
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            return img_str

    def __init__(self, camera: cv2.VideoCapture, calibration_needed=True):
        self.camera = camera
        self.l_eye = None
        self.r_eye = None
        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.solver = pnp_solver.PoseEstimator((height, width))
        self.calibrator = CameraSystemFactory(self.solver)
        self.screen = None
        self.head = None
        if calibration_needed:
            self.camera_calibration_server = self.CameraCalibrationServer(camera)
            zpc = zerorpc.Server(self.camera_calibration_server)
            self.camera_calibration_server.add_server(zpc)
            zpc.bind('tcp://127.0.0.1:4243')
            self.electron = subprocess.Popen(
                ["./frontend/node_modules/.bin/electron", "./frontend", "camera_calibrator"])
            zpc.run()

    def calibration_tick(self, time_now, predictor):
        img = [self.get_picture()]
        return self.calibrator.calibrate_remember(img, time_now, predictor)

    def calibration_corner_end(self, corner):
        self.calibrator.calibration_end(corner)

    def calibration_end(self):
        self.l_eye, self.r_eye, self.head = self.calibrator.calibration_final()

    def get_picture(self):
        ret, img = self.camera.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = PIL.Image.fromarray(img)
        return img

    def update_gazes_history(self, eye_one_vector, eye_two_vector, np_points, time_now):
        head_rotation, head_translation = self.head.add_history(np_points, time_now)
        world_to_camera = vector_to_camera_coordinate_system(head_rotation, head_translation)
        left_eye = sum(self.solver.model_points_68[36:41]) / 6
        right_eye = sum(self.solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)
        eye_one_screen = self.l_eye.add_history(eye_one_vector, time_now, right_eye)
        eye_two_screen = self.r_eye.add_history(eye_two_vector, time_now, left_eye)
        return eye_one_screen, eye_two_screen

    def vector_to_seen(self, eye_vector, eye_point=None):
        if eye_point is None:
            eye_point = [0, 0, 0]
        return utilities.get_pixel(eye_vector, eye_point)
