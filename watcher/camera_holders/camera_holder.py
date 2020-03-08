import base64
import signal
import subprocess
import typing
from io import BytesIO

import PIL
import cv2
import gevent
import numpy as np
import zerorpc
from PIL import Image

import predictor_module
from from_internet_or_for_from_internet import PNP_solver as PNP_solver
from utilities import get_world_to_camera_matrix
from .camera_system_factory import CameraSystemFactory


class CameraHolder:
    """This class is specified to store all logic and information about one camera

    :ivar _camera: (cv2.VideoCapture) real-world camera obj
    :ivar _l_eye: (Eye) used for remembering and smoothing eye gaze vector.
    :ivar _r_eye: (Eye) used for remembering and smoothing eye gaze vector.
    :ivar _solver: (from_internet_or_for_from_internet.PNP_solver.PoseEstimator)
                    used to store unrotated face position
    :ivar _factory: (CameraSystemFactory) used to create full camera system tick by tick.
    :ivar _screen: (Screen) used to match eye gaze vectors and onscreen target position
    :ivar _head: (Head) used for remembering and smoothing head position(rotation and translation)
    """

    def __init__(self, camera: cv2.VideoCapture, calibration_needed: bool = True):
        """The constructor of CameraHolder

        if calibration_needed is true then an electron app is run for calibrating the camera.

        :param camera: stores real-world camera object
        :param calibration_needed: flag that specifies whether camera calibration(brightness etc)
                                   should be used
        """
        self._camera = camera
        self._l_eye = None
        self._r_eye = None
        self._screen = None
        self._head = None

        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._solver = PNP_solver.PoseEstimator((height, width))

        self._factory = CameraSystemFactory(self._solver)

        if calibration_needed:
            self._camera_calibration_server = CameraCalibrationServer(camera)

            zpc = zerorpc.Server(self._camera_calibration_server)
            self._camera_calibration_server.add_server(zpc)

            zpc.bind('tcp://127.0.0.1:4243')
            self._electron = subprocess.Popen(
                ["./frontend/node_modules/.bin/electron", "./frontend", "camera_calibrator"])

            zpc.run()

    def calibration_tick(self, time_now: float,
                         predictor: predictor_module.BasicPredictor) -> str:
        """Just call a tick of _factory
            
        :param time_now: current time to remember
        :param predictor: which predictor to use
        :return: current head position and gazes
        """
        img = [self.get_picture()]
        return self._factory.calibrate_remember(img, time_now, predictor)

    def calibration_corner_end(self, corner: typing.Union[int, str]) -> None:
        """Just call a tick series end for one corner of _factory

        :param corner: corner_number, may be [1, 2...] or ["TL", "TR"...]
        :return: nothing is returned
        """
        self._factory.calibration_end(corner)

    def calibration_end(self) -> None:
        """Just call an end of _factory

        :return: nothing
        """
        self._l_eye, self._r_eye, self._head = self._factory.calibration_final()

    def get_picture(self) -> PIL.Image.Image:
        """Takes a photo with camera and turns it to grayscale"""
        ret, img = self._camera.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = PIL.Image.fromarray(img)
        return img

    def get_screen_positions(self,
                             eye_one_vector: np.ndarray(shape=(3,)),
                             eye_two_vector: np.ndarray(shape=(3,)),
                             np_points: np.ndarray(shape=(68, 2)),
                             time_now: float) -> (np.ndarray, np.ndarray):
        """Remembers current gazes and head pos and finds pixel gaze target.

        :param eye_one_vector: gaze vector from first eye
        :param eye_two_vector: gaze vector
        :param np_points: 68 face markers detected by dlib
        :param time_now: current time
        :return:
        """
        head_rotation, head_translation = self._head.get_smoothed_position(np_points, time_now)

        world_to_camera = get_world_to_camera_matrix(head_rotation, head_translation)

        left_eye = sum(self._solver.model_points_68[36:41]) / 6
        right_eye = sum(self._solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)

        eye_one_screen = self._l_eye.get_screen_point(eye_one_vector, time_now, right_eye)
        eye_two_screen = self._r_eye.get_screen_point(eye_two_vector, time_now, left_eye)
        return eye_one_screen, eye_two_screen


class CameraCalibrationServer:
    """This class is for camera calibration(brightness etc)

    To run it u have to mannually create and run zerorpc server etc(example in CameraHolder)

    :ivar _server: (zerorpc.Server) the server to communicate between electron and watcher
    :ivar _camera:
    :ivar _attributes:
    :ivar _attribute_names:

    """

    def __init__(self, camera: cv2.VideoCapture):
        """Contsruct object and check camera for parameter availability

        :param camera:
        """
        self._server = None
        self._camera = camera
        self._attributes = {}
        self._attribute_names = {
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

        for attr_name in self._attribute_names:
            i = self._attribute_names[attr_name]
            res = self._camera.get(i)
            if res != -1:
                self._attributes[i] = res, attr_name

    def add_server(self, server: zerorpc.Server) -> None:
        """Remember the zerorpc server and states its stop signal to gevent

        :param server:
        :return:
        """
        self._server = server
        gevent.signal(signal.SIGTERM, self._server.stop)

    def exit(self) -> None:
        """FOR RPC, Stop the zerorpc server"""
        self._server.stop()

    def get_attributes(self) -> typing.Dict[int, typing.Tuple[int, str]]:
        """FOR RPC, Get attributes and values

        :return:
        """
        return self._attributes

    def set_attribute(self,
                      attribute: typing.Union[str, int],
                      value: typing.Union[str, int]) -> None:
        """Set attribute value by its number

        :param attribute: attribute number, part of cv2.CAP_PROP...
        :param value: integer value
        :return:
        """
        attribute = int(attribute)
        value = int(value)
        self._camera.set(attribute, value)

    def get_frame(self) -> str:
        """Take a picture from cam, encode it to base64

        :return: (str) jpg picture in base64 encode
        """
        ret, image = self._camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        buffered = BytesIO()
        image = Image.fromarray(image)
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        return img_str
