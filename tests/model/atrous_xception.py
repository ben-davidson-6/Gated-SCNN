import unittest
import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.random.set_seed(1)
tf.config.set_visible_devices([], 'GPU')

import gated_shape_cnn.model.atrous_xception as atrous_xception


class TestAtrousXception(unittest.TestCase):
    def setUp(self):
        self.conv_name = 'conv2d_3'
        self.pool_name = 'block13_pool'
        self.fake_input = np.random.random([2, 2**7, 2**7, 3])
        with tf.device('/cpu:0'):
            x_input = tf.keras.layers.Input([2**7, 2**7, 3])
            x = tf.keras.layers.Conv2D(1, (2, 2), strides=(2, 2), name=self.conv_name)(x_input)
            x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), name=self.pool_name)(x)
            self.model = tf.keras.Model(x_input, x)

    def tearDown(self):
        del self.model

    def test_atrous_resolution(self):
        output = self.model(self.fake_input)
        atrous_xception.modify_layers(self.model)
        atrous_model = tf.keras.models.model_from_json(self.model.to_json())
        atrous_output = atrous_model(self.fake_input)

        # make sure the resolution is preserved
        self.assertEqual(output.shape[1], atrous_output.shape[1]//4)
        self.assertEqual(output.shape[2], atrous_output.shape[2]//4)

        # make sure the convolution is atrous
        atrous_rate = atrous_model.get_layer(self.conv_name).dilation_rate
        atrous_stride = atrous_model.get_layer(self.conv_name).strides
        atrous_padding = atrous_model.get_layer(self.conv_name).padding
        self.assertEqual((2, 2), atrous_rate)
        self.assertEqual((1, 1), atrous_stride)
        self.assertEqual('same', atrous_padding.lower())

        # make sure max pooling is the identity
        pool_stride = atrous_model.get_layer(self.pool_name).strides
        pool_size = atrous_model.get_layer(self.pool_name).pool_size
        pool_padding = atrous_model.get_layer(self.pool_name).padding
        self.assertEqual((1, 1), pool_size)
        self.assertEqual((1, 1), pool_stride)
        self.assertEqual('same', pool_padding.lower())

    def test_xception(self):
        model = atrous_xception.build_xception()
        self.assertEqual(type(model), tf.keras.Model)

        n = model.count_params()
        number_paramaters_for_xception = 20861480
        self.assertEqual(n, number_paramaters_for_xception)

        stride = 16
        original_shape = self.fake_input.shape
        output_shape = (original_shape[0], original_shape[1] // stride, original_shape[2] // stride, 2048)

        for mode in [False, True]:
            y = model(self.fake_input, training=mode)
            self.assertEqual(y.dtype, tf.float32)
            self.assertEqual(y.shape, output_shape)


if __name__ == '__main__':
    unittest.main()