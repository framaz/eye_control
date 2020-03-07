import PIL
import numpy as np

from .camera_holder import CameraHolder


class StubCameraHolder(CameraHolder):
    """Camera holder that releases the camera

    Used with gazeml predictor"""

    def __init__(self, camera):
        """Constructor, releases the camera"""
        super().__init__(camera, calibration_needed=False)
        camera.release()

    def get_picture(self):
        """Return a stub black picture"""
        img = PIL.Image.fromarray(np.zeros((720, 640)))
        return img
