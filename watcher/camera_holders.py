import cv2
import PIL
import calibrator
import predictor
import numpy as np

class CameraHolder:
    def __init__(self, camera: cv2.VideoCapture):
        self.camera = camera
        self.l_eye = None
        self.r_eye = None
        self.calibrator = calibrator.Calibrator()

    def configure(self):
        pass

    def calibration_tick(self, time_now):
        img = self.get_picture()
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


class Eye:
    def __init__(self):
        self.corner_vectors = np.zeros((len(calibrator.corner_dict), 3))
        self.translator = None
        self.point_vectors = np.zeros((len(calibrator.corner_dict), 2))
    def create_translator(self):
        self.corner_vectors = np.array(self.corner_vectors)
        self.point_vectors = np.array(predictor.pixel_func(self.corner_vectors))
        self.translator = calibrator.SeenToScreenTranslator(self.point_vectors)
