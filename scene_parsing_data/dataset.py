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

    def crop_size(self, image_shape):
        max_dim = tf.minimum(image_shape[0], image_shape[1])
        max_crop_size = tf.stack([max_dim, max_dim])
        reduction = tf.random.uniform(
            shape=[],
            minval=self.max_crop_downsample,
            maxval=1.)
        crop_size = max_crop_size*reduction
        crop_size = tf.cast(crop_size, tf.int32)
        return crop_size

    def crop_images(self, image, label, edge_label, train):
        image_shape = tf.shape(image)
        crop_size = self.crop_size(image_shape)
        all_input_tensor = tf.concat([image, label, edge_label], axis=-1)
        if train:
            cropped = tf.image.random_crop(all_input_tensor, crop_size)
            cropped = tf.image.random_flip_left_right(cropped)
        else:
            cropped = tf.image.central_crop(all_input_tensor, 1.)
        return cropped[..., :3], cropped[..., 3:4], cropped[..., 4:]

    def mold_to_network_input_shape(self, image, label, edge_label, train,):
        # todo check types
        image, label, edge_label = self.crop_images(image, label, edge_label, train=train)
        image = tf.image.resize(image, (self.network_input_h, self.network_input_w))
        label = tf.image.resize(label, (self.network_input_h, self.network_input_w), method='nearest')
        edge_label = tf.image.resize(edge_label, (self.network_input_h, self.network_input_w))
        return image, label, edge_label

    def colour_jitter(self, image,):
        image = tf.image.random_brightness(image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor)
        image = tf.image.random_saturation(image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor)
        image = tf.image.random_contrast(image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor)
        image = tf.image.random_hue(image, self.colour_aug_factor)
        return image

    def paths_to_tensors(self, im_path, label_path, edge_label_path, train):
        image = SceneParsingDataset.image_path_process(im_path)
        label = SceneParsingDataset.image_path_process(label_path)
        edge_label = SceneParsingDataset.image_path_process(edge_label_path)
        image, label, edge_label = self.mold_to_network_input_shape(image, label, edge_label, train=train)
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image, label, edge_label

    @staticmethod
    def get_paths(train):
        folders = scene_parsing_data.TRAINING_DIRS if train else scene_parsing_data.VALIDATION_DIRS
        image_paths = folders[scene_parsing_data.IMAGES]
        label_paths = [os.path.join(folders[scene_parsing_data.LABELS], x) for x in
                       os.listdir(folders[scene_parsing_data.LABELS])]
        edge_label_paths = [
            os.path.join(folders[scene_parsing_data.LABELS], scene_parsing_data.EDGE_PREFIX + x) for x in
            os.listdir(folders[scene_parsing_data.LABELS])]
        return image_paths, label_paths, edge_label_paths

    def process_batch(self, images, labels, edges, train):
        if train:
            images = self.colour_jitter(images)
        edges = tf.image.convert_image_dtype(edges, tf.float32)
        return images, labels, edges

    def build_dataset(self, train):
        image_paths, label_paths, edge_label_paths = SceneParsingDataset.get_paths(train)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths, edge_label_paths))
        dataset = dataset.shuffle(20000, seed=scene_parsing_data.SEED)
        dataset = dataset.map(
            lambda x, y, z: self.paths_to_tensors(x, y, z, train=train),
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