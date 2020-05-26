import tensorflow as tf


class Dataset:
    """
    All custom datasets should inherit from this class. To do so you need to provide two methods
        self.get_paths(train) -> image_paths, label_paths, edge_paths
        self.flat_to_one_hot(labels, edges) -> converts flat segmentations (h, w) to one_hot (h, w, c)
    """

    def __init__(
            self,
            n_classes,
            batch_size,
            network_input_h,
            network_input_w,
            max_crop_downsample,
            colour_aug_factor,
            debug,
            val_batch_size=2):
        """
        :param val_batch_size:
        :param batch_size:
        :param network_input_h height of training input:
        :param network_input_w width of training input:
        :param max_crop_downsample how far we will scale cropping window:
        :param colour_aug_factor:
        :param debug setting to true will give you a dataset with 1 element for both train and val:
        """
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.network_input_h = network_input_h
        self.network_input_w =  network_input_w
        self.max_crop_downsample = max_crop_downsample
        self.colour_aug_factor = colour_aug_factor
        self.debug = debug
        self.val_batch_size = val_batch_size

    @staticmethod
    def image_path_process(path):
        raw = tf.io.read_file(path,)
        image = tf.image.decode_image(raw, channels=3)
        image.set_shape([None, None, None])
        return image

    @staticmethod
    def label_path_process(path):
        raw = tf.io.read_file(path)
        label = tf.image.decode_png(raw, channels=1)
        return label

    def crop_size(self, all_input_shape):
        """

        :param all_input_shape [h, w, 3+c+2] the shape of every input with the channels concated:
        :return the size of the random crop:
        """
        # we can only crop as much as the smallest dimension
        max_size = tf.reduce_min(all_input_shape[:2])
        max_crop_size = tf.stack([max_size, max_size])

        # get random amount to scale by
        reduction = tf.random.uniform(
            shape=[],
            minval=self.max_crop_downsample,
            maxval=1.)
        crop_size = tf.cast(max_crop_size, tf.float32)*reduction
        crop_size = tf.cast(crop_size, tf.int32)

        crop = tf.concat([crop_size, all_input_shape[-1:]], axis=0)
        crop = tf.where(crop == 0, 1, crop)
        return crop

    def crop_images(self, image, label, edge_label):
        """
        :param image tensor:
        :param label tensor:
        :param edge_label tensor:
        :return cropped image, cropped label, cropped edge:

        randomly crop the data, with a random sized crop
        """

        # concat all inputs so we can crop them in the same random manner
        all_input_tensor = tf.concat([image, label, edge_label], axis=-1)
        tensor_shape = tf.shape(all_input_tensor)

        # random crop size
        crop_size = self.crop_size(tensor_shape)

        # crop the images
        cropped = tf.image.random_crop(all_input_tensor, crop_size)
        return cropped[..., :3], cropped[..., 3:4], cropped[..., 4:]

    def resize_images(self, image, label, edge_label):
        """
        :param image tensor:
        :param label tensor:
        :param edge_label tensor:
        :return resized data:

        resize data, for training all inputs are shaped (self.network_input_h, self.network_input_w)
        """
        image = tf.image.resize(image, (self.network_input_h, self.network_input_w))
        label = tf.image.resize(label, (self.network_input_h, self.network_input_w), method='nearest')
        edge_label = tf.image.resize(edge_label, (self.network_input_h, self.network_input_w), method='nearest')
        return image, label, edge_label

    def colour_jitter(self, image,):
        if self.colour_aug_factor == 0.:
            return image
        image = tf.image.random_brightness(image, self.colour_aug_factor)
        image = tf.image.random_saturation(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor)
        image = tf.image.random_contrast(
            image, 1. - self.colour_aug_factor, 1 + self.colour_aug_factor)
        image = tf.image.random_hue(image, self.colour_aug_factor)
        return image

    @staticmethod
    def paths_to_tensors(im_path, label_path, edge_label_path):
        """
        :param im_path:
        :param label_path:
        :param edge_label_path:
        :return image tensor [h, w, 3] tf.uint8; label [h, w, 1] tf.int32; edge [h, w, 1] tf.int32
        """
        image = Dataset.image_path_process(im_path)
        label = Dataset.label_path_process(label_path)
        edge_label = Dataset.label_path_process(edge_label_path)
        return image, label, edge_label

    @staticmethod
    def random_flip(image, label, edge):
        """random left right flips"""
        all_tensors = tf.concat([image, label, edge], axis=-1)
        all_tensors = tf.image.random_flip_left_right(all_tensors[None])[0]
        return all_tensors[..., :3], all_tensors[..., 3:4], all_tensors[..., 4:]

    def get_paths(self, train):
        raise NotImplementedError('you must implement this in sub class')

    def flat_to_one_hot(self, labels, edges):
        labels = tf.one_hot(labels[..., 0], self.n_classes)
        edges = tf.one_hot(edges[..., 0], 2)
        return labels, edges

    @staticmethod
    def validate_flat_to_one_hot(labels, edges):
        # make sure they are of shape [b, h, w, c]
        tf.debugging.assert_rank(labels, 4, 'label')
        tf.debugging.assert_rank(labels, 4, 'edges')

        # make sure have convincing number of channels
        tf.debugging.assert_shapes([
            (edges, ('b', 'h', 'w', 2)),
            (labels, ('b', 'h', 'w', 'c')),
        ])
        label_channels = tf.shape(labels)[-1]
        tf.assert_greater(label_channels, 1)

    def process_training_batch(self, images, labels, edges):
        """batch convert to one hot and apply colour jitter"""
        labels, edges = self.flat_to_one_hot(labels, edges)
        Dataset.validate_flat_to_one_hot(labels, edges)
        images = tf.cond(
            tf.greater(tf.random.uniform([]), 0.5),
            lambda: self.colour_jitter(images),
            lambda: images)
        return images, labels, edges

    def process_validation_batch(self, images, labels, edges):
        """batch convert to one hot and make the image float32"""
        labels, edges = self.flat_to_one_hot(labels, edges)
        images = tf.cast(images, tf.float32)
        return images, labels, edges

    def get_raw_tensor_dataset(self, train):
        """
        :param train bool which data split to get:
        :return a dataset of tensors [(im, label, edge), ...]:
        """

        # get the paths to the data
        image_paths, label_paths, edge_label_paths = self.get_paths(train=train)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths, edge_label_paths))
        if train:
            dataset = dataset.shuffle(3000)

        # convert the paths to tensors
        dataset = dataset.map(Dataset.paths_to_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def build_training_dataset(self):
        """
        training dataset
            - random left right flips
            - random crop locations
            - random crop sizes
            - all crops resized to (self.network_h, self.network_w)
                - has the effect of multiple scales
            - random colour jitter
        """
        # get dataset of tensors (im, label, edge)
        dataset = self.get_raw_tensor_dataset(train=True)

        # training augmentations
        dataset = dataset.map(Dataset.random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.crop_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # batch processing, convert to one hot, also apply colour jitter here
        # so we can do it on batch rather than per image
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.process_training_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self.debug:
            dataset = dataset.take(1)
        return dataset

    def build_validation_dataset(self):
        """
        val dataset:
            - full size images
            - no augmentations
            - fixed batch size of VAL_BATCH (=2)
        """
        # get dataset of tensors (im, label, edge)
        dataset = self.get_raw_tensor_dataset(train=False)

        # batch process
        dataset = dataset.batch(self.val_batch_size, drop_remainder=True)
        dataset = dataset.map(self.process_validation_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self.debug:
            dataset = dataset.take(1)
        return dataset


if __name__ == '__main__':
    pass