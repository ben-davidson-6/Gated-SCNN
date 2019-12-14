import tensorflow as tf
import scene_parsing_data


class SceneParsingDataset:

    def __init__(self, batch_size, network_input_h, network_input_2, max_crop_downsample):
        self.batch_size = batch_size
        self.network_input_h = network_input_h
        self.network_input_w =  network_input_2
        self.max_crop_downsample = max_crop_downsample

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

    def crop_images(self, image, label):
        image_shape = tf.shape(image)
        crop_size = self.crop_size(image_shape)
        all_input_tensor = tf.concat([image, tf.expand_dims(label, axis=-1)])
        cropped = tf.image.random_crop(all_input_tensor, crop_size)
        return cropped[..., :3], cropped[..., 3:]

    def mold_to_network_input_shape(self, image, label):
        image, label = self.crop_images(image, label)
        image = tf.image.resize(image, (self.network_input_h, self.network_input_w))
        label = tf.image.resize(label, (self.network_input_h, self.network_input_w), method='nearest')
        return image, label

    def colour_jitter(self, image,):
        image = tf.image.random_brightness(image, 1. - self.factor, 1 + self.factor)
        image = tf.image.random_saturation(image, 1. - self.factor, 1 + self.factor)
        image = tf.image.random_contrast(image, 1. - self.factor, 1 + self.factor)
        image = tf.image.random_hue(image, self.factor)
        return image

    def training_augmentations(self, image):
        image = self.colour_jitter(image)

    def paths_to_tensors(self, im_path, label_path):
        image = SceneParsingDataset.image_path_process(im_path)
        label = SceneParsingDataset.image_path_process(label_path)
        image, label = self.mold_to_network_input_shape(image, label)
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image, label

    def build_dataset(self, train):
        folders = scene_parsing_data.TRAINING_DIRS if train else scene_parsing_data.VALIDATION_DIRS
        image_paths = folders[scene_parsing_data.IMAGES]
        label_paths = folders[scene_parsing_data.LABELS]
    def build_training_dataset(self):
        return self.build_dataset(train=True)

    def build_validation_dataset(self):
        return self.build_dataset(train=False)