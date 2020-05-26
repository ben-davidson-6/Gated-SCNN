import unittest
import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.random.set_seed(1)
tf.config.set_visible_devices([], 'GPU')

import gated_shape_cnn.model.atrous_xception as atrous_xception


class TestAtrousXception(unittest.TestCase):
    def setUp(self):
        self.fake_input = np.random.random([2, 2**7, 2**7, 3])

    def tearDown(self):
        del self.fake_input

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