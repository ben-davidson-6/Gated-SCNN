import tensorflow as tf
import gated_shape_cnn.training.loss as gscnn_loss
import numpy as np
import random

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

    def test_gen_dice_in_range(self):
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

    def test_gen_dice_in_range_when_weird(self):
        # multiple classes at single point
        for i in range(20):
            for n_class in range(2, 3):
                logits = tf.random.uniform([1, 10, 10, n_class]) - 0.5
                image = tf.ones_like(logits)
                case = 'all class on, n_class {}, case {}'.format(n_class, i)
                with self.subTest(case):
                    l_val = gscnn_loss._generalised_dice(image, logits)
                    self.assertGreaterEqual(l_val, 0.)
                    self.assertLessEqual(l_val, 1.)

    def _gen_random_loss_input(self, i):
        mult_fact = i > 0
        shape = (random.randint(1, 8), random.randint(2, 20), random.randint(2, 20))
        n_class = random.randint(2, 10)
        loss_weights = tf.random.uniform([4], minval=0., maxval=20)
        case = 'shape {}, n_class {}, weight {}'.format(shape, n_class, loss_weights)
        gt_label = tf.random.uniform(
            shape,
            maxval=n_class,
            dtype=tf.int32)
        gt_label = tf.one_hot(gt_label, depth=n_class)*mult_fact
        logits = tf.random.uniform(shape + (n_class,)) - 0.5
        shape_head = tf.random.uniform(shape + (1,))
        if mult_fact == 0:
            edge_label = tf.ones(shape, dtype=tf.int32)
        else:
            edge_label = tf.random.uniform(
                shape,
                maxval=1,
                dtype=tf.int32)
        edge_label = tf.one_hot(edge_label, depth=2)

        return (gt_label, logits, shape_head, edge_label, loss_weights), case

    def test_loss_positive(self):
        for i in range(100):
            inputs, case = self._gen_random_loss_input(i)
            gt_label, logits, shape_head, edge_label, loss_weights = inputs
            with self.subTest(case):
                v = tf.add_n(
                    gscnn_loss.loss(
                        gt_label,
                        logits,
                        shape_head,
                        edge_label,
                        loss_weights=loss_weights))
                self.assertGreater(v, 0.)

    def test_loss_gradients(self):
        for i in range(100):
            inputs, case = self._gen_random_loss_input(i)
            gt_label, logits, shape_head, edge_label, loss_weights = inputs
            with self.subTest(case):
                with tf.GradientTape() as  tape:
                    tape.watch(logits)
                    loss = tf.add_n(
                        gscnn_loss.loss(
                            gt_label,
                            logits,
                            shape_head,
                            edge_label,
                            loss_weights=loss_weights))
                gradients = tape.gradient(loss, [logits])
                for g in gradients:
                    try:
                        tf.debugging.check_numerics(g, message='found nan in loss')
                    except:
                        self.fail('found nan')





