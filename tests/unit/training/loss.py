import tensorflow as tf
import unittest
import gated_shape_cnn.training.loss as gscnn_loss
import numpy as np


np.random.seed(1)
tf.random.set_seed(1)
tf.config.set_visible_devices([], 'GPU')


class TestLoss(tf.test.TestCase):

    def test_generalised_dice(self):

        # perfect answer 0 loss
        c_1 = tf.constant([1., 0.], tf.float32)
        c_2 = tf.constant([0., 1.], tf.float32)
        image = tf.stack([[c_1, c_2], [c_2, c_1]])[None]
        loss = gscnn_loss._generalised_dice(image, image, from_logits=False)
        self.assertEqual(loss, 0.)

        # perfect wrong answer max loss
        loss = gscnn_loss._generalised_dice(image, 1 - image, from_logits=False)
        self.assertEqual(loss, 1.)

        # loss always in [0, 1]
        for i in range(20):
            for n_class in range(2, 3):
                logits = tf.random.uniform([1, 10, 10, n_class]) - 0.5
                image = tf.random.uniform(
                    minval=0,
                    maxval=n_class,
                    shape=[1, 10, 10, 1],
                    dtype=tf.int32)
                image = tf.one_hot(image, depth=n_class)
                with self.subTest('n_class {}, case {}'.format(n_class, i)):
                    l_val = gscnn_loss._generalised_dice(image, logits)
                    self.assertGreaterEqual(l_val, 0.)
                    self.assertLessEqual(l_val, 1.)

    def test_edge_mag(self):
        pass





