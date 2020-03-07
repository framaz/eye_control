import numpy as np
import typing
from PIL import Image, ImageDraw, ImageFont

from .data_enhancer import DataEnhancer


class CornerTextEnhancer(DataEnhancer):
    """Enhancer to write text in left top corner

    :ivar _text_size: in pixels"""

    def __init__(self, text_size: int = 9, **kwargs):
        """Constructs obj

        :param text_size: in pixels
        :param kwargs:
        """
        self._text_size = text_size
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray,
                obj_to_out: object = "kek") -> typing.Tuple[Image.Image, dict]:
        """Apply enhancer - draw text in top left corner

        :param pic:
        :param np_points: facial landmarks
        :param obj_to_out: any object to out as a string
        :return:
        """
        pic, output = super().process(pic, np_points)

        text = str(obj_to_out)

        drawer = ImageDraw.Draw(pic)
        font = ImageFont.truetype("arial.ttf", size=self._text_size)
        drawer.multiline_text([0, 0], text, fill=(255, 0, 0), font=font)
        return pic, output
