import tensorflow as tf
import unittest
import gated_shape_cnn.training.utils as utils
import numpy as np


np.random.seed(1)
tf.random.set_seed(1)
tf.config.set_visible_devices([], 'GPU')


class TestValidation(tf.test.TestCase):
    def test_validate_image(self,):
        image = tf.random.uniform([1, 10, 10, 3], dtype=tf.float32)
        try:
            utils.validate_image_tensor(image)
        except (ValueError, TypeError):
            self.fail('validation said bad image but was good')

        image = tf.random.uniform([1, 10, 10, 4], dtype=tf.float32)
        self.assertRaises(ValueError, lambda: utils.validate_image_tensor((image)))

        image = tf.random.uniform([10, 10, 3], dtype=tf.float32)
        self.assertRaises(ValueError, lambda: utils.validate_image_tensor((image)))

        image = tf.random.uniform([1, 10, 10, 3], dtype=tf.float32)
        image = tf.cast(image, tf.uint8)
        self.assertRaises(TypeError, lambda: utils.validate_image_tensor((image)))

    def test_validate_label(self,):
        label = tf.random.uniform([1, 10, 10, 4], dtype=tf.float32)
        try:
            utils.validate_label_tensor(label)
        except (ValueError, TypeError):
            self.fail('validation said bad image but was good')

        label = tf.random.uniform([1, 10, 10, 1], dtype=tf.float32)
        self.assertRaises(ValueError, lambda: utils.validate_label_tensor((label)))

        label = tf.random.uniform([10, 10, 3], dtype=tf.float32)
        self.assertRaises(ValueError, lambda: utils.validate_label_tensor((label)))

        label = tf.random.uniform([1, 10, 10, 3], dtype=tf.float32)
        label = tf.cast(label, tf.uint8)
        self.assertRaises(TypeError, lambda: utils.validate_label_tensor((label)))

    def test_validate_edge(self,):
        edge = tf.random.uniform([1, 10, 10, 2], dtype=tf.float32)
        try:
            utils.validate_edge_tensor(edge)
        except (ValueError, TypeError):
            self.fail('validation said bad image but was good')

        edge = tf.random.uniform([1, 10, 10, 1], dtype=tf.float32)
        self.assertRaises(ValueError, lambda: utils.validate_edge_tensor((edge)))

        edge = tf.random.uniform([1, 10, 10, 3], dtype=tf.float32)
        self.assertRaises(ValueError, lambda: utils.validate_edge_tensor((edge)))

        edge = tf.random.uniform([10, 10, 2], dtype=tf.float32)
        self.assertRaises(ValueError, lambda: utils.validate_edge_tensor((edge)))

        edge = tf.random.uniform([1, 10, 10, 2], dtype=tf.float32)
        edge = tf.cast(edge, tf.uint8)
        self.assertRaises(TypeError, lambda: utils.validate_edge_tensor((edge)))


class TestEdgeBuilder(tf.test.TestCase):
    def test_label_to_one_hot(self):
        n_classes = 2
        eye = np.eye(n_classes)
        flat_array = np.array([[0, 0, 1]])
        one_hot = utils._label_to_one_hot_for_boundary(flat_array, n_classes=2)
        self.assertAllClose(one_hot, eye[flat_array])

        n_classes = 3
        eye = np.eye(n_classes)
        flat_array = np.array([[0, 0, 1]])
        one_hot = utils._label_to_one_hot_for_boundary(flat_array, n_classes=n_classes)
        self.assertAllClose(one_hot, eye[flat_array])

        n_classes = 2
        flat_array = np.array([[1, 4]])
        one_hot = utils._label_to_one_hot_for_boundary(flat_array, n_classes=n_classes)
        self.assertAllClose(one_hot, np.array([[[0., 1.], [0., 0.]]]))

    def test_flat_label_to_edge_label(self):
        n_classes = 2
        segmentation = np.array([
            [0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 1, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0,],
        ], dtype=np.int32)
        edge = np.array([
            [0, 0, 0, 1, 0, 0, 0, ],
            [0, 0, 1, 1, 1, 0, 0, ],
            [0, 1, 1, 1, 1, 1, 0, ],
            [0, 0, 1, 1, 1, 0, 0, ],
            [0, 0, 0, 1, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, ],
            [0, 0, 0, 0, 0, 0, 0, ],
        ], dtype=np.int32)
        edge = np.expand_dims(edge, axis=-1)

        built_edge = utils.flat_label_to_edge_label(segmentation, n_classes, radius=2)
        self.assertAllClose(edge, built_edge)

if __name__ == '__main__':
    unittest.main()