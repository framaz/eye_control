import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data_enhancer import DataEnhancer


class CornerTextEnhancer(DataEnhancer):
    def __init__(self, text_size=9, **kwargs):
        self.text_size = text_size
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray, obj_to_out="kek"):
        text = str(obj_to_out)
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        font = ImageFont.truetype("arial.ttf", size=self.text_size)
        drawer.multiline_text([0, 0], text, fill=(255, 0, 0), font=font)
        return pic, output