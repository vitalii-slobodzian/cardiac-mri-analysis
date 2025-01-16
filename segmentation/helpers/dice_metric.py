import torch
from torch import Tensor
from fastai.vision.all import Tensor, ItemTransform, PILImage, PILMask

class DiceMetric():
    def __init__(self, dice_smooth: float):
        self.dice_smooth = dice_smooth

    def multi_dice(self, input: Tensor, targs: Tensor, class_id=0, inverse=False) -> Tensor:
        n = targs.shape[0]
        input = input.argmax(dim=1).view(n, -1)
        # replace all with class_id with 1 all else with 0 to have binary case
        output = (input == class_id).float()
        # same for targs
        targs = (targs.view(n, -1) == class_id).float()
        if inverse:
            output = 1 - output
            targs = 1 - targs
        intersect = (output * targs).sum(dim=1).float()
        union = (output+targs).sum(dim=1).float()
        res = (2. * intersect + self.dice_smooth) / (union + self.dice_smooth)
        res[torch.isnan(res)] = 1
        return res.mean()

    def class_0(self, input, targs): return self.multi_dice(input, targs, class_id=0)
    def class_1(self, input, targs): return self.multi_dice(input, targs, class_id=1)
    def class_2(self, input, targs): return self.multi_dice(input, targs, class_id=2)
    def class_3(self, input, targs): return self.multi_dice(input, targs, class_id=3)
    def class_4(self, input, targs): return self.multi_dice(input, targs, class_id=4)
    def class_5(self, input, targs): return self.multi_dice(input, targs, class_id=5)
    def class_6(self, input, targs): return self.multi_dice(input, targs, class_id=6)
    def class_7(self, input, targs): return self.multi_dice(input, targs, class_id=7)
    def class_8(self, input, targs): return self.multi_dice(input, targs, class_id=8)
    def class_9(self, input, targs): return self.multi_dice(input, targs, class_id=9)


def get_y(file_path):
    return str(file_path).replace('images', 'masks')
