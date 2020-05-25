import tensorflow as tf
import gated_shape_cnn.training.dataset as dataset
import numpy as np
import random
import tempfile
import uuid
import imageio
import os

from pathlib import Path
from test.utils import *


np.random.seed(1)
tf.random.set_seed(1)
tf.config.set_visible_devices([], 'GPU')


class TestDataset(tf.test.TestCase):
    def setUp(self):
        self.dataset = dataset.Dataset(
            n_classes=2,
            batch_size=2,
            network_input_h=200,
            network_input_w=200,
            max_crop_downsample=0.5,
            colour_aug_factor=0.25,
            debug=False,)

    def test_image_path_process(self):
        im_path = Path(tempfile.gettempdir()).joinpath(uuid.uuid4().hex + '.png')
        a = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        imageio.imsave(im_path, a)
        # print(imageio.help('jpg'))
        try:
            b = self.dataset.image_path_process(str(im_path))
            self.assertAllClose(a, b)
        finally:
            os.remove(im_path)

    def test_label_path_process(self):
        im_path = Path(tempfile.gettempdir()).joinpath(uuid.uuid4().hex + '.png')
        a = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
        imageio.imsave(im_path, a)
        # print(imageio.help('jpg'))
        try:
            b = self.dataset.label_path_process(str(im_path))
            self.assertAllClose(a, b)
        finally:
            os.remove(im_path)

    def test_crop_size(self):
        # [h, w, 3 + c + 2]
        for _ in range(100):
            h = random.randint(1, 100)
            w = random.randint(1, 100)
            c = random.randint(1, 100)
            all_input_shape = (h, w, c)
            with self.subTest(all_input_shape):
                crop_size = self.dataset.crop_size(all_input_shape)
                c_h = crop_size[0]
                c_w = crop_size[1]
                self.assertGreater(c_h, 0)
                self.assertGreater(c_w, 0)
                self.assertLessEqual(c_h, h)
                self.assertLessEqual(c_w, w)
                self.assertEqual(c, crop_size[-1])

    def assert_image_smaller_or_equal(self, t_a, t_b):
        for dimension_a, dimension_b in zip(t_a.shape, t_b.shape):
            self.assertLessEqual(dimension_b, dimension_a)

    def test_crop_images(self):
        for _ in range(100):
            h = random.randint(1, 100)
            w = random.randint(1, 100)
            c = random.randint(2, 100)
            all_input_shape = (h, w, c)
            with self.subTest(all_input_shape):
                image = random_image(h, w)
                label = random_label(h, w, c, flat=False)
                edge = random_edge(h, w, flat=False)
                c_im, c_edge, c_lab = self.dataset.crop_images(image, label, edge)
                self.assert_image_smaller_or_equal(image, c_im)
                self.assert_image_smaller_or_equal(label, c_lab)
                self.assert_image_smaller_or_equal(edge, c_edge)

    def test_resize_images(self):
        out_shape = (self.dataset.network_input_h, self.dataset.network_input_w)
        for _ in range(100):
            h = random.randint(1, 100)
            w = random.randint(1, 100)
            c = random.randint(2, 100)
            all_input_shape = (h, w, c)
            with self.subTest(all_input_shape):
                image = random_image(h, w)
                label = random_label(h, w, c, flat=False)
                edge = random_edge(h, w, flat=False)
                c_im, c_edge, c_lab = self.dataset.resize_images(image, label, edge)

                self.assertEqual(out_shape, c_im.shape[:-1])
                self.assertEqual(out_shape, c_edge.shape[:-1])
                self.assertEqual(out_shape, c_lab.shape[:-1])

    def test_colour_jitter(self):

        h = random.randint(1, 100)
        w = random.randint(1, 100)
        with self.subTest('make sure random applied'):
            image = random_image(h, w)
            jittered_0 = self.dataset.colour_jitter(image)
            jittered_1 = self.dataset.colour_jitter(image)
            self.assertNotAllClose(jittered_0, jittered_1)

        with self.subTest('no jitter'):
            h = random.randint(1, 100)
            w = random.randint(1, 100)
            image = random_image(h, w)
            self.dataset.colour_aug_factor = 0.
            jittered = self.dataset.colour_jitter(image)
            self.assertAllClose(jittered, image)

