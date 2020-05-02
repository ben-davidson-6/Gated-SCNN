import tensorflow as tf
import os

import datasets.scene_parsing_data
import datasets.scene_parsing_data.raw_dataset

from gscnn.dataset import Dataset


class SceneParsing(Dataset):

    def __init__(
            self,
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug=False):
        super(SceneParsing, self).__init__(batch_size, network_input_h, network_input_w, max_crop_downsample, colour_aug_factor, debug)

    def get_paths(self, train):
        folders = datasets.scene_parsing_data.TRAINING_DIRS if train else datasets.scene_parsing_data.VALIDATION_DIRS
        image_dir = folders[datasets.scene_parsing_data.IMAGES]
        label_dir = folders[datasets.scene_parsing_data.LABELS]

        example_ids = []
        for x in os.listdir(image_dir):
            example_ids.append(x[:-4])

        image_paths = [os.path.join(image_dir, x + '.jpg') for x in example_ids]
        label_paths = [os.path.join(label_dir, x + '.png') for x in example_ids]
        edge_paths = [os.path.join(label_dir, datasets.scene_parsing_data.EDGE_PREFIX + x + '.png') for x in example_ids]

        return image_paths, label_paths, edge_paths

    def flat_to_one_hot(self, labels, edges):
        labels = tf.one_hot(labels[..., 0], datasets.scene_parsing_data.N_CLASSES)
        edges = tf.one_hot(edges[..., 0], 2)
        return labels, edges


if __name__ == '__main__':
    pass