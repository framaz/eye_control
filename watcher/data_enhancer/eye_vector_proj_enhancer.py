import typing
from PIL import ImageDraw
from PIL import Image
import numpy as np

from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from .data_enhancer import DataEnhancer


class EyeVectorProjEnhancer(DataEnhancer):
    """Draws projections of eye vectors on a 2d white surface"""

    def __init__(self, **kwargs):
        """Constructor"""
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray,
                eye_vector_left: np.ndarray = None,
                eye_vector_right: np.ndarray = None) -> typing.Tuple[Image.Image, dict]:
        """Apply - Draw projections...

        :param pic:
        :param np_points: facial landmarks
        :param eye_vector_left:
        :param eye_vector_right:
        :return:
        """
        if eye_vector_right is None:
            eye_vector_right = [0, 0, 1]
        if eye_vector_left is None:
            eye_vector_left = [0, 0, 1]

        pic, output = super().process(pic, np_points)

        drawer = ImageDraw.Draw(pic)
        drawer.rectangle([0, 0, 1920, 1080], fill=(255, 255, 255))
        middle = [960, 540]

        eye_vector_right *= 1000
        eye_vector_left *= 1000
        drawer.line([middle[0], middle[1], middle[0] + eye_vector_left[0],
                     middle[1] + eye_vector_left[2]], fill=(0, 255, 0))
        drawer.line([middle[0], middle[1], middle[0] + eye_vector_right[0],
                     middle[1] + eye_vector_right[2]], fill=(0, 255, 0))
        return pic, output
