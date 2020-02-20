from PIL import ImageDraw

from .data_enhancer import DataEnhancer


class PointDataEnhancer(DataEnhancer):
    def __init__(self, point_size=2, **kwargs):
        super().__init__(**kwargs)
        self.point_radius = point_size

    def process(self, pic, np_points):
        pic, output = super().process(pic, np_points)
        drawer = ImageDraw.Draw(pic)
        for [x, y] in np_points:
            drawer.ellipse([x - self.point_radius, y - self.point_radius, x + self.point_radius, y + self.point_radius],
                           fill=0)
        return pic, output