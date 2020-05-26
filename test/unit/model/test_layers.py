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


class TestGradMag(tf.test.TestCase):

    def test_correct_gradients(self):
        tensor = tf.constant([
            [0., 0, 1, 0., 0.],
            [0., 0, 1, 0., 0.],
        ])[None, :, :, None]
        out = gscnn_layers.gradient_mag(tensor)
        should_out = tf.constant([
            [0., 1, 0, 1., 0.],
            [0., 1, 0, 1., 0.],
        ])[None, :, :, None]
        self.assertEqual(out.get_shape(), tf.TensorShape([1, 2, 5, 1]))
        self.assertAllClose(out, should_out)

        tensor = tf.constant([
            [0., 0.],
            [0., 0.],
            [1., 1.],
            [0., 0.],
            [0., 0.],
        ])[None, :, :, None]
        out = gscnn_layers.gradient_mag(tensor)
        should_out = tf.constant([
            [0., 0.],
            [1., 1.],
            [0., 0.],
            [1., 1.],
            [0., 0.],
        ])[None, :, :, None]
        self.assertEqual(out.get_shape(), tf.TensorShape([1, 5, 2, 1]))
        self.assertAllClose(out, should_out)

    def test_empty_image(self):
        tensor = tf.constant([
            [0., 0, 0, 0., 0.],
            [0., 0, 0, 0., 0.],
        ])[None, :, :, None]
        out = gscnn_layers.gradient_mag(tensor)
        should_out = tf.constant([
            [0., 0, 0, 0., 0.],
            [0., 0, 0, 0., 0.],
        ])[None, :, :, None]
        self.assertEqual(out.get_shape(), tf.TensorShape([1, 2, 5, 1]))
        self.assertAllClose(out, should_out)


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

    def test_weights(self):
        # build model if not already
        self.gate_conv(self.tensor)
        variables = self.gate_conv.trainable_variables
        self.assertEqual(len(variables), 1 + 2 + 1 + 2)

    def tearDown(self):
        del self.gate_conv


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

    def test_weights(self):
        # build model if not already
        self.gated_shape_conv([self.tensor, self.tensor])
        variables = self.gated_shape_conv.trainable_variables
        self.assertEqual(len(variables), 6 + 2)

    def tearDown(self):
        del self.gated_shape_conv


class TestResnetPreact(tf.test.TestCase):
    def setUp(self):
        self.tensor = tf.random.uniform([1, 16, 16, 3])
        self.tensor_shape_tuple = (1, 16, 16, 3)
        self.tensor_shape = tf.TensorShape(self.tensor_shape_tuple)
        self.resnet_preact = gscnn_layers.ResnetPreactUnit()

    def test_output_shape(self):
        y = self.resnet_preact(self.tensor)
        self.assertEqual(y.get_shape(), self.tensor_shape)

    def test_batch_norm_doing_something(self):
        y_training = self.resnet_preact(self.tensor, training=True)
        y_inference = self.resnet_preact(self.tensor, training=False)
        self.assertNotAllClose(y_inference, y_training)

    def test_weights(self):
        # build model if not already
        self.resnet_preact(self.tensor)
        variables = self.resnet_preact.trainable_variables
        self.assertEqual(len(variables), 4)


class TestShapeAttention(tf.test.TestCase):
    def setUp(self):
        self.tensors = []
        self.tensor_shapes = []
        for i in range(4):
            shape_tuple = (1, 16//2**i, 16//2**i, i + 12)
            self.tensor_shapes.append(shape_tuple)
            self.tensors.append(tf.random.uniform(shape_tuple))
        self.shape_attention = gscnn_layers.ShapeAttention()

    def test_output_shape(self):
        y = self.shape_attention(self.tensors)
        self.assertEqual(y.get_shape(), self.tensor_shapes[0][:-1] + (1,))

    def test_batch_norm_doing_something(self):
        y_training = self.shape_attention(self.tensors, training=True)
        y_inference = self.shape_attention(self.tensors, training=False)
        self.assertNotAllClose(y_inference, y_training)

    def test_weights(self):
        # build model if not already
        self.shape_attention(self.tensors)
        variables = self.shape_attention.trainable_variables
        self.assertEqual(len(variables), 4*3 + 8*3 + 6*2 + 1)


class TestShapeStream(tf.test.TestCase):
    def setUp(self):
        self.tensors = []
        self.tensor_shapes = []
        for i in range(4):
            shape_tuple = (1, 16//2**i, 16//2**i, i + 12)
            self.tensor_shapes.append(shape_tuple)
            self.tensors.append(tf.random.uniform(shape_tuple))
        self.image_edge = tf.random.uniform([1, 23, 23, 1])
        self.shape_stream = gscnn_layers.ShapeStream()

    def test_output_shape(self):
        attention_map, edge_out = self.shape_stream((self.tensors, self.image_edge))
        self.assertEqual(attention_map.get_shape(), self.tensor_shapes[0][:-1] + (1,))
        self.assertEqual(edge_out.get_shape(), self.tensor_shapes[0][:-1] + (1,))

    def test_batch_norm_doing_something(self):
        y_training = self.shape_stream((self.tensors, self.image_edge), training=True)
        y_inference = self.shape_stream((self.tensors, self.image_edge), training=False)
        self.assertNotAllClose(y_inference, y_training)

    def test_weights(self):
        # build model if not already
        self.shape_stream((self.tensors, self.image_edge))
        variables = self.shape_stream.trainable_variables
        self.assertEqual(len(variables), 49 + 1)


class TestAtrousConvolution(tf.test.TestCase):
    def setUp(self):
        self.tensor_shape = (2, 16, 16, 5)
        self.tensor = tf.random.uniform(self.tensor_shape)

    def test_output_shape(self):
        c_out = 12
        for rate in [1, 2, 3]:
            for kernel_size in [1, 2, 3]:
                with self.subTest('kernel {}, rate {}'.format(kernel_size, rate)):
                    conv = gscnn_layers.AtrousConvolution(rate=1, filters=c_out, kernel_size=2)
                    out = conv(self.tensor)
                    self.assertEqual(out.get_shape(), self.tensor_shape[:-1] + (c_out,))

    def test_rate_bigger_than_tensor(self):
        c_out = 12
        conv = gscnn_layers.AtrousConvolution(rate=100, filters=c_out, kernel_size=2)
        out = conv(self.tensor)
        self.assertEqual(out.get_shape(), self.tensor_shape[:-1] + (c_out,))

    def test_weights(self):
        # build model if not already
        conv = gscnn_layers.AtrousConvolution(rate=1, filters=1, kernel_size=2)
        conv(self.tensor)
        variables = conv.trainable_variables
        self.assertEqual(len(variables), 2)


class TestAtrousSpatialPyramid(tf.test.TestCase):
    def setUp(self):
        self.backbone_head_shape = (2, 16, 16, 5)
        self.backbone_head = tf.random.uniform(self.backbone_head_shape)
        self.shape_head_shape = (2, 32, 32, 1)
        self.shape_head = tf.random.uniform(self.shape_head_shape)
        self.intermediate_shape = (2, 64, 64, 5)
        self.intermediate = tf.random.uniform(self.intermediate_shape)
        self.aspp = gscnn_layers.AtrousPyramidPooling(out_channels=256)

    def test_output_shape(self):
        for out_channels in [8, 16]:
            with self.subTest('out_channels {}'.format(out_channels)):
                out = self.aspp((self.backbone_head, self.shape_head, self.intermediate))
                self.assertEqual(out.get_shape(), self.intermediate_shape[:-1] + (304,))

    def test_batch_norm_doing_something(self):
        y_training = self.aspp((self.backbone_head, self.shape_head, self.intermediate), training=True)
        y_inference = self.aspp((self.backbone_head, self.shape_head, self.intermediate), training=False)
        self.assertNotAllClose(y_inference, y_training)


class TestFinalLogitLayer(tf.test.TestCase):
    def setUp(self):
        self.tensor_shape = (2, 16, 16, 5)
        self.tensor = tf.random.uniform(self.tensor_shape)
        self.num_classes = 10
        self.logit_layer = gscnn_layers.FinalLogitLayer(num_classes=self.num_classes)

    def test_output_shape(self):
        out = self.logit_layer(self.tensor)
        self.assertEqual(out.get_shape(), self.tensor_shape[:-1] + (self.num_classes,))

    def test_batch_norm_doing_something(self):
        y_training = self.logit_layer(self.tensor, training=True)
        y_inference = self.logit_layer(self.tensor, training=False)
        self.assertNotAllClose(y_inference, y_training)

    def tearDown(self):
        del self.logit_layer


class TestXceptionBackbone(tf.test.TestCase):
    def setUp(self):
        self.tensor_shape = (2, 16, 16, 3)
        self.tensor = tf.random.uniform(self.tensor_shape)

    def test_output_shape(self):
        xception = gscnn_layers.XceptionBackbone()
        out = xception(self.tensor)
        self.assertEqual(len(out), 4)
        self.assertEqual(type(out), dict)

    def test_batch_norm_doing_something(self):
        xception = gscnn_layers.XceptionBackbone()
        y_training = xception(self.tensor, training=True)
        y_inference = xception(self.tensor, training=False)
        self.assertNotAllClose(y_inference, y_training)


if __name__ == '__main__':
    unittest.main()