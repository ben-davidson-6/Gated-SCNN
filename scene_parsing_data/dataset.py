import tensorflow as tf
import scene_parsing_data
import os


class SceneParsingDataset:

    def __init__(self, batch_size, network_input_h, network_input_2, max_crop_downsample, colour_aug_factor):
        self.batch_size = batch_size
        self.network_input_h = network_input_h
        self.network_input_w =  network_input_2
        self.max_crop_downsample = max_crop_downsample
        self.colour_aug_factor = colour_aug_factor

    @staticmethod
    def image_path_process(path):
        raw = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        return image

    @staticmethod
    def label_path_process(path):
        raw = tf.io.read_file(path)
        label = tf.image.decode_png(raw, channels=1)
        return label

    def crop_size(self, all_input_shape):
        max_dim = tf.minimum(all_input_shape[0], all_input_shape[1])
        max_crop_size = tf.stack([max_dim, max_dim])
        reduction = tf.random.uniform(
            shape=[],
            minval=self.max_crop_downsample,
            maxval=1.,
            seed=scene_parsing_data.SEED)
        crop_size = tf.cast(max_crop_size, tf.float32)*reduction
        crop_size = tf.cast(crop_size, tf.int32)
        return tf.concat([crop_size, all_input_shape[-1:]], axis=0)

    def crop_images(self, image, label, edge_label, train):
        all_input_tensor = tf.concat([image, label, edge_label], axis=-1)
        tensor_shape = tf.shape(all_input_tensor)
        crop_size = self.crop_size(tensor_shape)
        if train:
            cropped = tf.image.random_crop(all_input_tensor, crop_size, seed=scene_parsing_data.SEED)
            cropped = tf.image.random_flip_left_right(cropped, seed=scene_parsing_data.SEED)
        else:
            cropped = tf.image.central_crop(all_input_tensor, 1.)
        return cropped[..., :3], cropped[..., 3:4], cropped[..., 4:]

    def mold_to_network_input_shape(self, image, label, edge_label, train,):
        image, label, edge_label = self.crop_images(image, label, edge_label, train=train)

        # image becomes float here
        image = tf.image.resize(image, (self.network_input_h, self.network_input_w))
        label = tf.image.resize(label, (self.network_input_h, self.network_input_w), method='nearest')
        edge_label = tf.image.resize(edge_label, (self.network_input_h, self.network_input_w), method='nearest')

        return image, label, edge_label

    def colour_jitter(self, image,):
        image = tf.image.random_brightness(
            image, self.colour_aug_factor, seed=scene_parsing_data.SEED)
        image = tf.image.random_saturation(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor, seed=scene_parsing_data.SEED)
        image = tf.image.random_contrast(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor, seed=scene_parsing_data.SEED)
        image = tf.image.random_hue(image, self.colour_aug_factor, seed=scene_parsing_data.SEED)
        return image

    @staticmethod
    def paths_to_tensors(im_path, label_path, edge_label_path):
        image = SceneParsingDataset.image_path_process(im_path)
        label = SceneParsingDataset.label_path_process(label_path)
        edge_label = SceneParsingDataset.label_path_process(edge_label_path)
        return image, label, edge_label

    @staticmethod
    def get_paths(train):
        folders = scene_parsing_data.TRAINING_DIRS if train else scene_parsing_data.VALIDATION_DIRS
        image_dir = folders[scene_parsing_data.IMAGES]
        label_dir = folders[scene_parsing_data.LABELS]

        example_ids = []
        for x in os.listdir(image_dir):
            example_ids.append(x[:-4])

        image_paths = [os.path.join(image_dir, x + '.jpg') for x in example_ids]
        label_paths = [os.path.join(label_dir, x + '.png') for x in example_ids]
        edge_paths = [os.path.join(label_dir, scene_parsing_data.EDGE_PREFIX + x + '.png') for x in example_ids]

        return image_paths, label_paths, edge_paths

    def process_batch(self, images, labels, edges, train):
        if train:
            images = self.colour_jitter(images)
        # labels have a single ending dimension we need to kill
        # for one hot to work properly
        labels = tf.one_hot(labels[..., 0], scene_parsing_data.N_CLASSES)
        edges = tf.one_hot(edges[..., 0], 2)
        return images, labels, edges

    def build_dataset(self, train):
        image_paths, label_paths, edge_label_paths = SceneParsingDataset.get_paths(train)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths, edge_label_paths))
        if train:
            dataset = dataset.shuffle(20000, seed=scene_parsing_data.SEED)
        dataset = dataset.map(
            SceneParsingDataset.paths_to_tensors,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y, z: self.mold_to_network_input_shape(x, y, z, train=train),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(
            lambda x, y, z: self.process_batch(x, y, z, train),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def build_training_dataset(self):
        return self.build_dataset(train=True)

    def build_validation_dataset(self):
        return self.build_dataset(train=False)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import scene_parsing_data.utils as utils


    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    s = SceneParsingDataset(1, 128, 128, 0.75, 0.25)

    for im, label, edge in s.build_training_dataset():
        im = im.numpy()[0].astype(np.uint8)
        label = np.argmax(label.numpy()[0, ...], axis=-1)
        label, _ = utils.flat_label_to_plottable(label)
        edge = edge.numpy()[0, ..., 1]
        plt.subplot(3, 1, 1)
        plt.imshow(im)
        plt.subplot(3, 1, 2)
        plt.imshow(label)
        plt.subplot(3, 1, 3)
        plt.imshow(edge)
        plt.show()
        break
