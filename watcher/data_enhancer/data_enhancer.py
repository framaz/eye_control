import numpy as np
import typing
from PIL import Image


class DataEnhancer:
    """Basic data enhancer class

    Every data enhancer is first created and then is applied"""
    def __init__(self, **kwargs):
        """Construct object"""
        pass

    def process(self, pic: Image.Image, np_points: np.ndarray) -> typing.Tuple[Image.Image, dict]:
        """Apply enhancer - just transform to Image.Image and convert to RGB

        :param pic:
        :param np_points: facial landmarks
        :return:
        """
        # shouldn't be called from basic DataEnhancer
        if type(self) == DataEnhancer:
            raise NotImplementedError("AbsDataEnhancer process call")

        if isinstance(pic, np.ndarray):
            pic = pic.astype(dtype=np.uint8)
            pic = Image.fromarray(pic)

        pic = pic.convert('RGB')
        output = {}
        return pic, output
