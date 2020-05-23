import unittest
import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.random.set_seed(1)
tf.config.set_visible_devices([], 'GPU')

import gated_shape_cnn.model.layers as gscnn_layers


class TestResize(unittest.TestCase):

    def test_resize_by_size(self):
        tensor = tf.random.uniform([2, 10, 10, 3])

        # check makes bigger
        h = 11
        w = 11
        out_t = gscnn_layers.resize_to(tensor, target_shape=(h, w))
        self.assertEqual(out_t.get_shape(), tf.TensorShape((2, h, w, 3)))
        h = 12
        w = 11
        out_t = gscnn_layers.resize_to(tensor, target_shape=(h, w))
        self.assertEqual(out_t.get_shape(), tf.TensorShape((2, h, w, 3)))

        # smaller
        h = 9
        w = 8
        out_t = gscnn_layers.resize_to(tensor, target_shape=(h, w))
        self.assertEqual(out_t.get_shape(), tf.TensorShape((2, h, w, 3)))

        # same
        h = 10
        w = 10
        out_t = gscnn_layers.resize_to(tensor, target_shape=(h, w))
        self.assertEqual(out_t.get_shape(), tf.TensorShape((2, h, w, 3)))

    def test_resize_by_tensor(self):
        tensor = tf.random.uniform([2, 10, 10, 3])

        # check makes bigger
        target_t = tf.random.uniform([1, 90, 90, 2])
        out_t = gscnn_layers.resize_to(tensor, target_t=target_t)
        self.assertEqual(out_t.get_shape()[1:-1], target_t.get_shape()[1:-1])


class TestGateConv(tf.test.TestCase):
    def setUp(self):
        self.tensor = tf.random.uniform([1, 16, 16, 3])
        self.tensor_shape_tuple = (1, 16, 16, 3)
        self.tensor_shape = tf.TensorShape((1, 16, 16, 3))
        self.gate_conv = gscnn_layers.GateConv()

    def test_output_shape(self):
        y = self.gate_conv(self.tensor)
        self.assertEqual(y.get_shape(), tf.TensorShape([1, 16, 16, 1]))
        self.assertEqual(self.gate_conv.compute_output_shape(self.tensor_shape), (1, 16, 16, 1))

    def test_batch_norm_doing_something(self):
        y_training = self.gate_conv(self.tensor, training=True)
        y_inference = self.gate_conv(self.tensor, training=False)
        self.assertNotAllClose(y_inference, y_training)


class TestGatedShapeConv(tf.test.TestCase):
    def setUp(self):
        self.tensor = tf.random.uniform([1, 16, 16, 3])
        self.tensor_shape_tuple = (1, 16, 16, 3)
        self.tensor_shape = tf.TensorShape(self.tensor_shape_tuple)
        self.gated_shape_conv = gscnn_layers.GatedShapeConv()

    def test_output_shape(self):
        y = self.gated_shape_conv([self.tensor, self.tensor])
        self.assertEqual(y.get_shape(), self.tensor_shape)

    def test_batch_norm_doing_something(self):
        y_training = self.gated_shape_conv([self.tensor, self.tensor], training=True)
        y_inference = self.gated_shape_conv([self.tensor, self.tensor], training=False)
        self.assertNotAllClose(y_inference, y_training)



if __name__ == '__main__':
    unittest.main()