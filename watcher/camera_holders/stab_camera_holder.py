import PIL
import numpy as np

from .camera_holder import CameraHolder


class StabCameraHolder(CameraHolder):
    def __init__(self, camera):
        super().__init__(camera, calibration_needed=False)
        camera.release()
    def get_picture(self):
        img = PIL.Image.fromarray(np.zeros((720, 640)))
        return img