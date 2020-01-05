import tensorflow as tf
import cityscapes.raw_dataset
import cityscapes


class CityScapes:

    def __init__(self, batch_size, network_input_h, network_input_2, max_crop_downsample, colour_aug_factor, data_dir, mixup_val, merge_labels=False):
        self.batch_size = batch_size
        self.network_input_h = network_input_h
        self.network_input_w =  network_input_2
        self.max_crop_downsample = max_crop_downsample
        self.colour_aug_factor = colour_aug_factor
        self.mixup_val = mixup_val
        self.raw_data = cityscapes.raw_dataset.CityScapesRaw(data_dir)
        self.merge_labels = merge_labels

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
        max_crop_size = tf.stack([1024, 1024])
        reduction = tf.random.uniform(
            shape=[],
            minval=self.max_crop_downsample,
            maxval=1.,
            seed=cityscapes.SEED)
        crop_size = tf.cast(max_crop_size, tf.float32)*reduction
        crop_size = tf.cast(crop_size, tf.int32)
        return tf.concat([crop_size, all_input_shape[-1:]], axis=0)

    def crop_images(self, image, label, edge_label, train):
        all_input_tensor = tf.concat([image, label, edge_label], axis=-1)
        tensor_shape = tf.shape(all_input_tensor)
        crop_size = self.crop_size(tensor_shape)
        if train:
            cropped = tf.image.random_crop(all_input_tensor, crop_size, seed=cityscapes.SEED)
            cropped = tf.image.random_flip_left_right(cropped, seed=cityscapes.SEED)
        else:
            cropped = tf.image.central_crop(all_input_tensor, 1.0)
        return cropped[..., :3], cropped[..., 3:4], cropped[..., 4:]

    def mixup(self, image, label, edge_label):

        t = tf.concat([image, label, edge_label], axis=-1)

        beta = tf.compat.v1.distributions.Beta(self.mixup_val, self.mixup_val, )
        input_tensor_reverse = tf.reverse(t, [0])
        lambdas = beta.sample(tf.shape(image)[0])
        input_tensors = lambdas[:, None, None, None] * t + (
                    1 - lambdas[:, None, None, None]) * input_tensor_reverse
        return input_tensors[..., :3], input_tensors[..., 3: 3 + cityscapes.N_CLASSES], input_tensors[..., -2:]

    def mold_to_network_input_shape(self, image, label, edge_label, train,):
        image, label, edge_label = self.crop_images(image, label, edge_label, train=train)

        # image becomes float here
        image = tf.image.resize(image, (self.network_input_h, self.network_input_w))
        label = tf.image.resize(label, (self.network_input_h, self.network_input_w), method='nearest')
        edge_label = tf.image.resize(edge_label, (self.network_input_h, self.network_input_w), method='nearest')

        return image, label, edge_label

    def colour_jitter(self, image,):
        image = tf.image.random_brightness(
            image, self.colour_aug_factor, seed=cityscapes.SEED)
        image = tf.image.random_saturation(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor, seed=cityscapes.SEED)
        image = tf.image.random_contrast(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor, seed=cityscapes.SEED)
        image = tf.image.random_hue(image, self.colour_aug_factor, seed=cityscapes.SEED)
        return image

    @staticmethod
    def paths_to_tensors(im_path, label_path, edge_label_path):
        image = CityScapes.image_path_process(im_path)
        label = CityScapes.label_path_process(label_path)
        edge_label = CityScapes.label_path_process(edge_label_path)
        return image, label, edge_label

    def get_paths(self, train):
        split = cityscapes.TRAIN if train else cityscapes.VAL
        paths = self.raw_data.dataset_paths(split)
        image_paths, label_paths, edge_paths = zip(*paths)
        return list(image_paths), list(label_paths), list(edge_paths)

    def process_batch(self, images, labels, edges, train):
        # labels have a single ending dimension we need to kill
        # for one hot to work properly
        labels = tf.one_hot(labels[..., 0], cityscapes.N_CLASSES)
        edges = tf.one_hot(edges[..., 0], 2)
        if train:
            images = self.colour_jitter(images)
            if self.mixup_val is not None:
                images, labels, edges = self.mixup(images, labels, edges)
        else:
            images = tf.cast(images, tf.float32)
        if self.merge_labels:
            return images, tf.concat([labels, edges], axis=-1)
        else:
            return images, labels, edges

    def build_training_dataset(self):
        image_paths, label_paths, edge_label_paths = self.get_paths(train=True)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths, edge_label_paths))
        dataset = dataset.shuffle(20000, seed=cityscapes.SEED)
        dataset = dataset.map(
            CityScapes.paths_to_tensors,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y, z: self.mold_to_network_input_shape(x, y, z, train=True),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(
            lambda x, y, z: self.process_batch(x, y, z, train=True),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def build_validation_dataset(self):
        image_paths, label_paths, edge_label_paths = self.get_paths(train=False)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths, edge_label_paths))
        dataset = dataset.map(
            CityScapes.paths_to_tensors,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1)
        dataset = dataset.map(
            lambda x, y, z: self.process_batch(x, y, z, train=False),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__ == '__main__':
    pass