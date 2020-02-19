import base64
import collections
import copy
import signal
import subprocess
from io import BytesIO

import cv2
import PIL
import gevent
import zerorpc
from PIL import Image

import calibrator
import plane_by_eye_vectors
import predictor_module
import numpy as np
import matplotlib.pyplot as plt
import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import seen_to_screen

SHOW_EYE_HISTORY_AND_BOARDERS = False


def vector_to_camera_coordinate_system(rotation, translation):
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[0:3, 0:3] = rotation_matrix
    matrix[:, 3] = translation.reshape((3,))
    return matrix


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
        self.calibrator = calibrator.Calibrator(self.solver)
        self.screen = None
        self.head = None
        if calibration_needed:
            self.camera_calibration_server = self.CameraCalibrationServer(camera)
            zpc = zerorpc.Server(self.camera_calibration_server)
            self.camera_calibration_server.add_server(zpc)
            zpc.bind('tcp://127.0.0.1:4243')
            self.electron = subprocess.Popen(["./frontend/node_modules/.bin/electron", "./frontend", "camera_calibrator"])
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
        return self.screen.get_pixel(eye_vector, eye_point)

class StabCameraHolder(CameraHolder):
    def __init__(self, camera):
        super().__init__(camera, calibration_needed=False)
        camera.release()
    def get_picture(self):
        img = PIL.Image.fromarray(np.zeros((1280, 720)))
        return img

class Eye:
    def __init__(self, eye_type):
        assert eye_type == 'l' or eye_type == 'r'
        self.corner_vectors = np.zeros((len(calibrator.corner_dict), 3))
        self.translator = None
        self.corner_points = np.zeros((len(calibrator.corner_dict), 2))
        self.history = collections.deque()
        self.screen = None
        self.tmp_value = 0
        self.eye_type = eye_type

    def create_translator(self, screen, eye_center):
        self.screen = screen
        self.corner_vectors = np.array(self.corner_vectors)
        for i in range(self.corner_vectors.shape[0]):
            self.corner_points[i] = screen.get_pixel(self.corner_vectors[i], eye_center)
        self.translator = seen_to_screen.SeenToScreenTranslator(self.corner_points)
        self.translator = self.translator

    def add_screen(self, screen):
        self.screen = screen

    def add_history(self, vector, time_now, eye_center):
        # TODO normal eye center detection
        vector = predictor_module.normalize(vector)
        self.tmp_value += 1
        if self.tmp_value >= 10 and SHOW_EYE_HISTORY_AND_BOARDERS:
            plt.plot(self.corner_points[:, 0], self.corner_points[:, 1], )
            np_history = np.array(self.history)
            plt.plot(np_history[:, 0], np_history[:, 1], )
            plt.show()
            self.tmp_value = 0
        res_pixel = self.screen.get_pixel(vector, eye_center)
        point = res_pixel
        self.history.append([point[0], point[1], time_now])
        seen_point = calibrator.smooth_n_cut(self.history, time_now)
        screen_point = self.translator.seen_to_screen(seen_point)
        return screen_point


def get_pixel(tens, a, b, c, d,
              width=0.54, heigth=0.30375,
              pixel_width=1920, pixel_heigth=1080):
    """k = -d / (a * x + b * y + c * z)
    print(x[0].item())
    x = k * x
    y = k * y
    z = k * z
    pixel_x = x / width * pixel_width
    pixel_y = y / heigth * pixel_heigth
    return pixel_x, pixel_y """
    return tens


class Screen:
    def __init__(self, l_eye: Eye, r_eye: Eye, solver, head_rotation, head_translation):
        # TODO non-dumb screen surface creation
        assert len(l_eye.corner_points) == 5
        assert len(r_eye.corner_points) == 5
        self.solver = solver
        pairs_list = []
        kek = []
        world_to_camera = vector_to_camera_coordinate_system(head_rotation, head_translation)
        for point in solver.model_points_68:
            kek.append(world_to_camera @ [*point, 1])
        kek = np.array(kek).transpose()
        left_eye = sum(self.solver.model_points_68[36:41]) / 6
        right_eye = sum(self.solver.model_points_68[42:47]) / 6
        left_eye = np.array([*left_eye, 1])
        left_eye = np.matmul(world_to_camera, left_eye)
        right_eye = np.array([*right_eye, 1])
        right_eye = np.matmul(world_to_camera, right_eye)
        for i in range(len(l_eye.corner_vectors)):
            pair = {'left': [left_eye, l_eye.corner_vectors[i]], 'right': [right_eye, r_eye.corner_vectors[i]]}
            pairs_list.append(pair)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*kek)
        colors = ['#ff0000', '#00ff00', '#0000ff', '#000000', '#ffffff']
        length = 1800
        for screen_point, color in zip(pairs_list, colors):
            arr = np.array(screen_point['left'])
            arr[1] = arr[1] * length + arr[0]
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=color)
            arr = np.array(screen_point['right'])
            arr[1] = arr[1] * length + arr[0]
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], c=color)
     #   self.a, self.b, self.c, self.d = plane_by_eye_vectors.get_plane_by_eye_vectors(pairs_list)
        self.a, self.b, self.c, self.d = 0, 0, 1, -450

        normal_vector = sum(l_eye.corner_vectors) + sum(r_eye.corner_vectors)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        [a, b, c] = normal_vector
        eye_middle = (left_eye + right_eye) / 2
        screen_point = eye_middle + length * normal_vector
        d = - normal_vector @ screen_point
        self.a, self.b, self.c, self.d = a, b, c, d

        def draw_plot(plane, axis, min_x=-200, min_y=-200, max_x=205, max_y=205):
            a, b, c, d = plane
            xx, yy = np.meshgrid(range(min_x, max_x + 5, max_x - min_x), range(min_y, max_y + 5, max_y - min_y))
            z = -(a * xx + b * yy + d) / c
            axis.plot_wireframe(xx, yy, z)
        draw_plot([a, b, c, d], ax)
        #plt.show()
        self.width = 0.54
        self.heigth = 0.30375
        self.pixel_width = 1920
        self.pixel_heigth = 1080

    def get_pixel(self, gaze_vector, eye_point=None):
        if eye_point is None:
            eye_point = [0, 0, 0]
        [xv, yv, zv] = gaze_vector
        [x0, y0, z0] = eye_point
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        t = - (a * x0 + b * y0 + c * z0 + d) / (a * xv + b * yv + c * zv)
        x = t * xv + x0
        y = t * yv + y0
        z = t * zv + z0

        pixel_x = x
        pixel_y = y
        return [pixel_x, pixel_y]


class Head:
    def __init__(self, starting_rotation_vector, starting_translation: np.ndarray, solver: pnp_solver):
        self.solver = solver
        self.start_rotation = starting_rotation_vector
        self.start_translation = starting_translation
        self.rotation_history = collections.deque()
        self.translation_history = collections.deque()

    def add_history(self, np_points, time_now):
        rotation, translation = self.solver.solve_pose(np_points)
        rotation = rotation.reshape((3,))
        translation = translation.reshape((3,))
        self.rotation_history.append([*rotation, time_now])
        self.translation_history.append([*translation, time_now])
        smoothed_rotation = calibrator.smooth_n_cut(self.rotation_history, time_now)
        smoothed_translation = calibrator.smooth_n_cut(self.translation_history, time_now)
        return smoothed_rotation, smoothed_translation

    def translate_vector_to_first_position(self, starting_rotation_vector, starting_translation):
        to_new_world = get_world_to_camera_matrix(self.solver, starting_rotation_vector, starting_translation,
                                                  is_vector_translation=True)


def get_world_to_camera_matrix(solver, starting_rotation_vector, starting_translation, is_vector_translation=False):
    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[0:3, 0:3], _ = cv2.Rodrigues(starting_rotation_vector)
    if not is_vector_translation:
        matrix[:, 3] = starting_translation.reshape((3,))
    return np.matmul(solver.camera_matrix, matrix)


def project_point(point, solver, starting_rotation_vector, starting_translation, is_vector_translation=False):
    """matrix = get_world_to_camera_matrix(solver, starting_rotation_vector, starting_translation, is_vector_translation=False)
    res = np.matmul(matrix, [*point, 1.])
    res0 = np.matmul(matrix, [0., 0., 0., 1.])
    res -= res0
    res = res[0: 2] / res[2]
    """
    res, _ = cv2.projectPoints(point, starting_rotation_vector, starting_translation, solver.camera_matrix,
                               solver.dist_coeefs)
    if is_vector_translation:
        res0, _ = cv2.projectPoints(np.array([0., 0., 0.]), starting_rotation_vector, starting_translation,
                                    solver.camera_matrix, solver.dist_coeefs)
        res = res.reshape((2,)) - res0.reshape((2,))
    return res.reshape((2,))
    """starting_rotation_matrix, _ = cv2.Rodrigues(starting_rotation_vector)
    t_matrix = np.zeros((4, 4))
    t_matrix[:3, :3] = starting_rotation_matrix
    if not is_vector_translation:
        t_matrix[0:3, 3] = starting_translation.reshape((3))
    else:
        t_matrix[0:3, 3] = [0, 0, 0]
    t_matrix[3] = np.array([0, 0, 0, 1])
    diag = np.diag([1, 1, 1, 1])[:3, :]
    matrix = np.matmul(diag, t_matrix)
    camera_matrix = solver.camera_matrix
    matrix = np.matmul(camera_matrix, matrix)
    return matrix"""
