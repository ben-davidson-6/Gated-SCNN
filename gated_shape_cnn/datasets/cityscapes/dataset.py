import tensorflow as tf

import gated_shape_cnn.datasets.cityscapes
import gated_shape_cnn.datasets.cityscapes.raw_dataset

from gated_shape_cnn.training import Dataset


class CityScapes(Dataset):

    def __init__(
            self,
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug,
            data_dir):
        super(CityScapes, self).__init__(
            gated_shape_cnn.datasets.cityscapes.N_CLASSES,
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug)
        self.raw_data = gated_shape_cnn.datasets.cityscapes.raw_dataset.CityScapesRaw(data_dir)

    def get_paths(self, train):
        """
        :param train:
        :return image_paths, label_paths, edge_paths:
            image_path[0] -> path to image 0
            label_paths[0] -> path to semantic seg of image 0
            edge_paths[0] -> path to edge seg of label 0
        """
        split = gated_shape_cnn.datasets.cityscapes.TRAIN if train else gated_shape_cnn.datasets.cityscapes.VAL
        paths = self.raw_data.dataset_paths(split)
        image_paths, label_paths, edge_paths = zip(*paths)
        return list(image_paths), list(label_paths), list(edge_paths)


if __name__ == '__main__':
    pass