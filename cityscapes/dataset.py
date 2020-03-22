import tensorflow as tf
import cityscapes.raw_dataset
import cityscapes

from gscnn.dataset import Dataset


class CityScapes(Dataset):

    def __init__(
            self,
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            data_dir, ):
        super(CityScapes, self).__init__(batch_size, network_input_h, network_input_w, max_crop_downsample, colour_aug_factor,)
        self.raw_data = cityscapes.raw_dataset.CityScapesRaw(data_dir)

    def get_paths(self, train):
        split = cityscapes.TRAIN if train else cityscapes.VAL
        paths = self.raw_data.dataset_paths(split)
        image_paths, label_paths, edge_paths = zip(*paths)
        return list(image_paths), list(label_paths), list(edge_paths)

    def flat_to_one_hot(self, labels, edges):
        labels = tf.one_hot(labels[..., 0], cityscapes.N_CLASSES)
        edges = tf.one_hot(edges[..., 0], 2)
        return labels, edges



if __name__ == '__main__':
    pass