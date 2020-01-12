from math import sqrt

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import head_tracker

class AbsDataEnhancer:
    def __init__(self, **kwargs):
        pass

    def process(self, pic, np_points):
        if type(self) == AbsDataEnhancer:
            raise NotImplementedError("AbsDataEnhancer process call")
        if not isinstance(pic, Image.Image):
            pic = Image.fromarray(pic)
        pic = pic.convert('RGB')
        output = {}
        return pic, output


class PointDataEnhancer(AbsDataEnhancer):
    def __init__(self, point_size=2, **kwargs):
        super().__init__(**kwargs)
        self.point_radius = point_size

    def process(self, pic, np_points):
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        for [x, y] in np_points:
            drawer.ellipse([x - self.point_radius, y - self.point_radius, x + self.point_radius, y + self.point_radius], fill=0)
        return pic, output


class WidthHeightDataEnhancer(PointDataEnhancer):
    def __init__(self, line_width=2, text_size=9, **kwargs):
        self.text_size = text_size
        self.line_width = line_width
        super().__init__(**kwargs)

    def process(self, pic: Image.Image, np_points: np.ndarray):
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        font = ImageFont.truetype("arial.ttf", size=self.text_size)
        for first, second in head_tracker.width_metrics+head_tracker.heigth_metrics:
            first = np_points[first]
            second = np_points[second]
            drawer.line([*first, *second], width=self.line_width)
            [text_x, text_y] = second
            text_x += 2
            val = sqrt((first[0]-second[0])**2 + (first[1]-second[1])**2)
            drawer.text([text_x, text_y], str(int(val)), font=font, fill=(255, 0, 0))
        return pic, output

if __name__ == "__main__":
    lel = WidthHeightDataEnhancer(kek=20)
    ImageFont.truetype("arial.ttf", size=8)
    lel = lel
