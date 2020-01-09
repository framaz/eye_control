import collections

import cv2
import PIL
import calibrator
import predictor
import numpy as np
import matplotlib.pyplot as plt

SHOW_EYE_HISTORY_AND_BOARDERS = False


class CameraHolder:
    def __init__(self, camera: cv2.VideoCapture):
        self.camera = camera
        self.l_eye = None
        self.r_eye = None
        self.calibrator = calibrator.Calibrator()
        self.screen = None

    def configure(self):
        pass

    def calibration_tick(self, time_now):
        img = [self.get_picture()]
        return self.calibrator.calibrate_remember(img, time_now)

    def calibration_corner_end(self, corner):
        self.calibrator.calibration_end(corner)

    def calibration_end(self):
        self.l_eye, self.r_eye = self.calibrator.calibration_final()

    def get_picture(self):
        ret, img = self.camera.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = PIL.Image.fromarray(img)
        return img

    def update_gazes_history(self, eye_one_vector, eye_two_vector, time_now):
        eye_one_screen = self.l_eye.add_history(eye_one_vector, time_now)
        eye_two_screen = self.r_eye.add_history(eye_two_vector, time_now)
        return eye_one_screen, eye_two_screen

    def vector_to_seen(self, eye_vector, eye_point=None):
        if eye_point is None:
            eye_point = [0, 0, 0]
        return self.screen.get_pixel(eye_vector, eye_point)


class Eye:
    def __init__(self, eye_type):
        assert eye_type == 'l' or eye_type == 'r'
        self.corner_vectors = np.zeros((len(calibrator.corner_dict), 3))
        self.translator = None
        self.point_vectors = np.zeros((len(calibrator.corner_dict), 2))
        self.history = collections.deque()
        self.screen = None
        self.tmp_value = 0
        self.eye_type = eye_type

    def create_translator(self, screen, eye_distance=1):
        self.screen = screen
        self.corner_vectors = np.array(self.corner_vectors)
        for i in range(self.corner_vectors.shape[0]):
            if self.eye_type == 'l':
                self.point_vectors[i] = screen.get_pixel(self.corner_vectors[i], [0, 0, 0])
            else:
                self.point_vectors[i] = screen.get_pixel(self.corner_vectors[i], [0, 1, 0])

        self.translator = calibrator.SeenToScreenTranslator(self.point_vectors)
        self.translator = self.translator

    def add_screen(self, screen):
        self.screen = screen

    def add_history(self, vector, time_now, eye_coords=None):
        if eye_coords is None:
            eye_coords = [0, 0, 0]
        if self.eye_type == 'l':
            eye_point = [0, 0, 0]
        else:
            eye_point = [0, 1, 0]
        for i in range(3):
            eye_point[i] += eye_coords[i]
        vector = predictor.normalize(vector)
        self.tmp_value += 1
        if self.tmp_value >= 10 and SHOW_EYE_HISTORY_AND_BOARDERS == True:
            plt.plot(self.point_vectors[:, 0], self.point_vectors[:, 1], )
            np_history = np.array(self.history)
            plt.plot(np_history[:, 0], np_history[:, 1], )
            plt.show()
            self.tmp_value = 0
        res_pixel = self.screen.get_pixel(vector, eye_point)
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
    def __init__(self, l_eye=None, r_eye=None):
        # TODO proper screen coords creation
        self.a = 0
        self.b = 0
        self.c = 1
        self.d = -1
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
        width = self.width
        heigth = self.heigth
        pixel_width = self.pixel_width
        pixel_heigth = self.pixel_heigth
        t = - (a * x0 + b * y0 + c * z0 + d) / (a * xv + b * yv + c * zv)
        x = t * xv + x0
        y = t * yv + y0
        pixel_x = x / width * pixel_width
        pixel_y = y / heigth * pixel_heigth
        return [pixel_x, pixel_y]
