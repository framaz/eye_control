import typing
from PIL import ImageDraw, Image
import numpy as np

from .data_enhancer import DataEnhancer


class PointDataEnhancer(DataEnhancer):
    """Draws facial landmarks

    :ivar point_radius: (int), in pixels
    """
    def __init__(self, point_size: int = 2, **kwargs):
        """Constructor

        :param point_size:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self._point_radius = point_size

    def process(self, pic: Image.Image, np_points: np.ndarray) -> typing.Tuple[Image.Image, dict]:
        """Apply - just draw landmarks

        :param pic:
        :param np_points: facial landmarks
        :return:
        """
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        for [x, y] in np_points:
            drawer.ellipse([x - self._point_radius, y - self._point_radius,
                            x + self._point_radius, y + self._point_radius],
                           fill=0)
        return pic, output
