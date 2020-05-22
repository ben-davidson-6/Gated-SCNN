import tensorflow as tf

import gated_shape_cnn.datasets.cityscapes
import gated_shape_cnn.datasets.cityscapes.raw_dataset

from gated_shape_cnn.training.dataset import Dataset


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
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug)
        self.raw_data = gated_shape_cnn.datasets.cityscapes.raw_dataset.CityScapesRaw(data_dir)

    def get_paths(self, train):
        split = gated_shape_cnn.datasets.cityscapes.TRAIN if train else gated_shape_cnn.datasets.cityscapes.VAL
        paths = self.raw_data.dataset_paths(split)
        image_paths, label_paths, edge_paths = zip(*paths)
        return list(image_paths), list(label_paths), list(edge_paths)

    def flat_to_one_hot(self, labels, edges):
        labels = tf.one_hot(labels[..., 0], gated_shape_cnn.datasets.cityscapes.N_CLASSES)
        edges = tf.one_hot(edges[..., 0], 2)
        return labels, edges


if __name__ == '__main__':
    pass