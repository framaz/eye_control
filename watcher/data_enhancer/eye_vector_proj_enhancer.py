from PIL import ImageDraw

from from_internet_or_for_from_internet import PNP_solver as pnp_solver
from .data_enhancer import DataEnhancer


class EyeVectorProjEnhancer(DataEnhancer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, pic, np_points, eye_vector_left, eye_vector_right):
        pic, output = super().process(pic, np_points)
        solver = pnp_solver.PoseEstimator((720, 1080))
        drawer = ImageDraw.Draw(pic)
        drawer.rectangle([0, 0, 1920, 1080], fill=(255, 255, 255))
        middle = [960, 540]
        eye_vector_right *= 1000
        eye_vector_left *= 1000
        drawer.line([middle[0], middle[1], middle[0] + eye_vector_left[0], middle[1] + eye_vector_left[2]], fill=(0,255,0))
        drawer.line([middle[0], middle[1], middle[0] + eye_vector_right[0], middle[1] + eye_vector_right[2]], fill=(0,255,0))
        return pic, output