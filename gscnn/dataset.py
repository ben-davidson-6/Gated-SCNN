import tensorflow as tf
import random


class Dataset:

    def __init__(
            self,
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug):
        self.batch_size = batch_size
        self.network_input_h = network_input_h
        self.network_input_w =  network_input_w
        self.max_crop_downsample = max_crop_downsample
        self.colour_aug_factor = colour_aug_factor
        self.debug = debug

    @staticmethod
    def image_path_process(path):
        raw = tf.io.read_file(path)
        image = tf.image.decode_png(raw, channels=3)
        return image

    @staticmethod
    def label_path_process(path):
        raw = tf.io.read_file(path)
        label = tf.image.decode_png(raw, channels=1)
        return label

    def crop_size(self, all_input_shape):
        max_crop_size = tf.stack([all_input_shape[0], all_input_shape[0]])
        reduction = tf.random.uniform(
            shape=[],
            minval=self.max_crop_downsample,
            maxval=1.)
        crop_size = tf.cast(max_crop_size, tf.float32)*reduction
        crop_size = tf.cast(crop_size, tf.int32)
        return tf.concat([crop_size, all_input_shape[-1:]], axis=0)

    def crop_images(self, image, label, edge_label, random_crop=True):
        all_input_tensor = tf.concat([image, label, edge_label], axis=-1)
        tensor_shape = tf.shape(all_input_tensor)
        crop_size = self.crop_size(tensor_shape)
        if random_crop:
            cropped = tf.image.random_crop(all_input_tensor, crop_size)
        else:
            cropped = tf.image.central_crop(all_input_tensor, 1.0)
        return cropped[..., :3], cropped[..., 3:4], cropped[..., 4:]

    def resize_images(self, image, label, edge_label):
        image = tf.image.resize(image, (self.network_input_h, self.network_input_w))
        label = tf.image.resize(label, (self.network_input_h, self.network_input_w), method='nearest')
        edge_label = tf.image.resize(edge_label, (self.network_input_h, self.network_input_w), method='nearest')
        return image, label, edge_label

    def colour_jitter(self, image,):
        image = tf.image.random_brightness(image, self.colour_aug_factor)
        image = tf.image.random_saturation(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor)
        image = tf.image.random_contrast(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor)
        image = tf.image.random_hue(image, self.colour_aug_factor)
        return image

    @staticmethod
    def paths_to_tensors(im_path, label_path, edge_label_path):
        image = Dataset.image_path_process(im_path)
        label = Dataset.label_path_process(label_path)
        edge_label = Dataset.label_path_process(edge_label_path)
        return image, label, edge_label

    @staticmethod
    def random_flip(image, label, edge):
        all_tensors = tf.concat([image, label, edge], axis=-1)
        all_tensors = tf.image.random_flip_left_right(all_tensors[None])[0]
        return all_tensors[..., :3], all_tensors[..., 3:4], all_tensors[..., 4:]

    def get_paths(self, train):
        raise NotImplementedError('you must implement this in sub class')

    def flat_to_one_hot(self, labels, edges):
        raise NotImplementedError('You must implement this in sub class')

    def process_training_batch(self, images, labels, edges):
        labels, edges = self.flat_to_one_hot(labels, edges)
        if random.random() > 0.5:
            images = self.colour_jitter(images)
        return images, labels, edges

    def process_validation_batch(self, images, labels, edges):
        labels, edges = self.flat_to_one_hot(labels, edges)
        images = tf.cast(images, tf.float32)
        return images, labels, edges

    def get_raw_tensor_dataset(self, train):
        image_paths, label_paths, edge_label_paths = self.get_paths(train=train)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths, edge_label_paths))
        if train:
            dataset = dataset.shuffle(3000)
        dataset = dataset.map(Dataset.paths_to_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def build_training_dataset(self):
        dataset = self.get_raw_tensor_dataset(train=True)
        dataset = dataset.map(Dataset.random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y, z: self.crop_images(x, y, z, random_crop=True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.process_training_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if self.debug:
            dataset = dataset.take(1)
        return dataset

    def build_validation_dataset(self):
        dataset = self.get_raw_tensor_dataset(train=False)
        dataset = dataset.batch(2, drop_remainder=True)
        dataset = dataset.map(self.process_validation_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        if self.debug:
            dataset = dataset.take(1)
        return dataset


if __name__ == '__main__':
    pass