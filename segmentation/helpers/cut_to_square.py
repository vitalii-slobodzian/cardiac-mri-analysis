
from fastai.vision.all import Tensor, ItemTransform, PILImage, PILMask

class CutToSquare(ItemTransform):
    def encodes(self, x):
        if isinstance(x, (list, tuple, dict, set)) and len(x) > 1:
            image, mask = x
            return PILImage.create(image), PILMask.create(mask)
        else:
            return x
