import numpy as np
from PIL import Image


class DataEnhancer:
    def __init__(self, **kwargs):
        pass

    def process(self, pic, np_points):
        if type(self) == DataEnhancer:
            raise NotImplementedError("AbsDataEnhancer process call")

        if isinstance(pic, np.ndarray):
            pic = pic.astype(dtype=np.uint8)
            pic = Image.fromarray(pic)
        pic = pic.convert('RGB')
        output = {}
        return pic, output
