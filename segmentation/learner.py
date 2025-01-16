from fastai_segmentation.utils import *
from fastai.vision.all import ImageBlock, MaskBlock, get_image_files, resnet34, resnet50
from fastai.data.block import DataBlock
from fastai.vision.learner import unet_learner
from fastai.data.transforms import RandomSplitter

def get_y_bssfp(file_path):
    return str(file_path).replace('bssfp', 'masks')

class Learner():
    def __init__(self, config):
        self.config = config

    def learn(self, loss_function, transforms):
        classes = self.config['classes']
        data_block = DataBlock(
            blocks=(ImageBlock,  MaskBlock(codes=classes)),
            get_items=get_image_files,
            get_y=get_y,
            splitter=RandomSplitter(),
            item_tfms=transforms
        )

        dice_helper = DiceMetric(1e-6)
        metrics = [getattr(dice_helper, f'class_' + str(i)) for i in range(1, len(classes))]

        dataloaders = data_block.dataloaders(
            self.config['dataset_path'],
            bs = self.config['batch_size'],
            num_workers = self.config['num_workers']
        )
        learn = unet_learner(
            dataloaders,
            resnet34,
            loss_func=loss_function,
            metrics=metrics
        )

        # learn.lr_find()
        learn.fit_one_cycle(self.config['epoch_number'], lr_max=1e-3)
        learn.unfreeze()

        # learn.lr_find()
        learn.fit_one_cycle(self.config['epoch_number'], lr_max=1e-4)

        return learn

    def export_model(self, model):
        model_output_path = self.config["model_output_path"]
        model_name = self.config["model_name"]

        formatted_time = datetime.now().strftime('%d_%m_%Y_%H_%M')
        model_path = os.path.join(model_output_path, f'{model_name}_{formatted_time}')
        model_name = f'{model_name}.pkl'

        os.makedirs(model_path)
        model.export(os.path.join(model_path, model_name))

    def show_results(model):
        model.show_results(max_n=4, figsize=(5, 10))
        result = model.validate()
        print(result)

