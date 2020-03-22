import tensorflow as tf
from gscnn.atrous_inception import build_inception
from gscnn.sync_norm import BatchNormalization


def resize_to(x, target_t=None, target_shape=None):
    if target_shape is None:
        s = tf.shape(target_t)
        target_shape = tf.stack([s[1], s[2]])

    return tf.image.resize(x, target_shape)


class GateConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GateConv, self).__init__(**kwargs)
        self.batch_norm_1 = BatchNormalization()
        self.conv_1 = None
        self.relu = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv2D(1, kernel_size=1, use_bias=False)
        self.batch_norm_2 = BatchNormalization()
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(in_channels, kernel_size=1)

    def call(self, x, training=None):
        x = self.batch_norm_1(x, training=training)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x, training=training)
        x = self.sigmoid(x)
        return x


class GatedShapeConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GatedShapeConv, self).__init__(**kwargs)
        self.conv_1 = None
        self.gated_conv = GateConv()

    def build(self, input_shape):
        feature_channels = input_shape[0][-1]
        self.conv_1 = tf.keras.layers.Conv2D(feature_channels, 1)

    def call(self, x, training=None):
        feature_map, shape_map = x
        features = tf.concat([feature_map, shape_map], axis=-1)
        alpha = self.gated_conv(features, training=training)
        gated = feature_map*(alpha + 1.)
        return self.conv_1(gated)


class ResnetPreactUnit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ResnetPreactUnit, self).__init__(**kwargs)
        self.bn_1 = BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv_1 = None
        self.bn_2 = BatchNormalization()
        self.conv_2 = None

    def build(self, input_shape):
        cs = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(cs, 3, padding='SAME', use_bias=False)
        self.conv_2 = tf.keras.layers.Conv2D(cs, 3, padding='SAME', use_bias=False)

    def call(self, x, training=None):
        shortcut = x
        x = self.bn_1(x, training)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x, training)
        x = self.relu(x)
        x = self.conv_2(x)
        return x + shortcut


class ShapeAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ShapeAttention, self).__init__(**kwargs)

        self.gated_conv_1 = GatedShapeConv()
        self.gated_conv_2 = GatedShapeConv()
        self.gated_conv_3 = GatedShapeConv()

        self.shape_reduction_2 = tf.keras.layers.Conv2D(1, 1)
        self.shape_reduction_3 = tf.keras.layers.Conv2D(1, 1)
        self.shape_reduction_4 = tf.keras.layers.Conv2D(1, 1)

        self.res_1 = ResnetPreactUnit()
        self.res_2 = ResnetPreactUnit()
        self.res_3 = ResnetPreactUnit()

        self.reduction_conv_1 = tf.keras.layers.Conv2D(32, 1)
        self.reduction_conv_2 = tf.keras.layers.Conv2D(16, 1)
        self.reduction_conv_3 = tf.keras.layers.Conv2D(8, 1)
        self.reduction_conv_4 = tf.keras.layers.Conv2D(1, 1, use_bias=False)
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, x, training=None):
        (s1, s2, s3, s4), shape = x
        s2 = self.shape_reduction_2(s2)
        s3 = self.shape_reduction_3(s3)
        s4 = self.shape_reduction_4(s4)

        x = self.res_1(s1, training=training)
        x = self.reduction_conv_1(x)
        s2 = resize_to(s2, target_t=x)
        x = self.gated_conv_1([x, s2], training=training)

        x = self.res_2(x, training=training)
        x = self.reduction_conv_2(x)
        s3 = resize_to(s3, target_t=x)
        x = self.gated_conv_2([x, s3], training=training)

        x = self.res_3(x, training=training)
        x = self.reduction_conv_3(x)
        s4 = resize_to(s4, target_t=x)
        x = self.gated_conv_3([x, s4], training=training)

        x = self.reduction_conv_4(x)
        x = self.sigmoid(x)

        return x


class ShapeStream(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ShapeStream, self).__init__(**kwargs)
        self.shape_attention = ShapeAttention()
        self.reduction_conv = tf.keras.layers.Conv2D(1, 1, use_bias=False, )
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, x, training=None):
        (shape_backbone_activations, image_edges), shape = x
        edge_out = self.shape_attention([shape_backbone_activations, shape], training=training)
        image_edges = resize_to(image_edges, target_t=edge_out)
        backbone_representation = tf.concat([edge_out, image_edges], axis=-1)
        shape_logits = self.reduction_conv(backbone_representation)
        shape_attention = self.sigmoid(shape_logits)
        return shape_attention, edge_out


class AtrousConvolution(tf.keras.layers.Layer):
    def __init__(self, rate, filters, kernel_size, **kwargs):
        super(AtrousConvolution, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.out_channels = filters
        self.rate = rate
        self.kernel = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=[self.kernel_size, self.kernel_size, in_channels, self.out_channels],
            initializer=tf.keras.initializers.GlorotNormal())

    def call(self, x, training=None):
        return tf.nn.atrous_conv2d(x, self.kernel, self.rate, padding='SAME')


class AtrousPyramidPooling(tf.keras.layers.Layer):

    def __init__(self, out_channels, **kwargs):
        super(AtrousPyramidPooling, self).__init__(**kwargs)
        self.relu = tf.keras.layers.ReLU()

        # for final output of backbone
        self.bn_1 = BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(out_channels, 1, use_bias=False)

        self.bn_2 = BatchNormalization()
        self.atrous_conv_1 = AtrousConvolution(6, filters=out_channels, kernel_size=3)

        self.bn_3 = BatchNormalization()
        self.atrous_conv_2 = AtrousConvolution(12, filters=out_channels, kernel_size=3)

        self.bn_4 = BatchNormalization()
        self.atrous_conv_3 = AtrousConvolution(18, filters=out_channels, kernel_size=3)

        # for backbone features
        self.bn_img = BatchNormalization()
        self.conv_img = tf.keras.layers.Conv2D(out_channels, 1, use_bias=False)

        # for shape features
        self.bn_shape = BatchNormalization()
        self.conv_shape = tf.keras.layers.Conv2D(out_channels, 1, use_bias=False)

        # 1x1 reduction convolutions
        self.conv_reduction_1 = tf.keras.layers.Conv2D(64, 1, use_bias=False)
        self.conv_reduction_2 = tf.keras.layers.Conv2D(256, 1, use_bias=False)

    def call(self, x, training=None):
        image_features, shape_features, intermediate_rep = x

        backbone_shape, intermediate_shape = tf.shape(image_features), tf.shape(intermediate_rep)
        backbone_shape = tf.stack([backbone_shape[1], backbone_shape[2]])
        intermediate_shape = tf.stack([intermediate_shape[1], intermediate_shape[2]])

        # process backbone features and the shape activations
        # from the shape stream
        img_net = tf.reduce_mean(image_features, axis=[1, 2], keepdims=True)
        img_net = self.conv_img(img_net)
        img_net = self.bn_img(img_net, training=training)
        img_net = self.relu(img_net)
        img_net = tf.image.resize(img_net, backbone_shape)

        shape_net = self.conv_shape(shape_features)
        shape_net = self.bn_shape(shape_net, training=training)
        shape_net = self.relu(shape_net)
        shape_net = tf.image.resize(shape_net, backbone_shape)

        net = tf.concat([img_net, shape_net], axis=-1)

        # process with atrous
        w = self.conv_1(image_features)
        w = self.bn_1(w, training=training)
        w = self.relu(w)

        x = self.atrous_conv_1(image_features)
        x = self.bn_2(x, training=training)
        x = self.relu(x)

        y = self.atrous_conv_2(image_features)
        y = self.bn_3(y, training=training)
        y = self.relu(y)

        z = self.atrous_conv_3(image_features)
        z = self.bn_4(z, training=training)
        z = self.relu(z)

        # atrous output from final layer of backbone
        # and shape stream

        net = tf.concat([net, w, x, y, z], axis=-1)
        net = self.conv_reduction_1(net)

        # combine intermediate representation
        intermediate_rep = self.conv_reduction_2(intermediate_rep)
        net = tf.image.resize(net, intermediate_shape)
        net = tf.concat([net, intermediate_rep], axis=-1)

        return net


class FinalLogitLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(FinalLogitLayer, self).__init__(**kwargs)
        self.bn_1 = BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(256, 3, padding='SAME', use_bias=False, activation=tf.nn.relu)
        self.bn_2 = BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(256, 3, padding='SAME', use_bias=False, activation=tf.nn.relu)
        self.bn_3 = BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv2D(num_classes, 1, padding='SAME', use_bias=False)

    def call(self, x, training=None):
        x = self.bn_1(x, training=training)
        x = self.conv_1(x)
        x = self.bn_2(x, training=training)
        x = self.conv_2(x)
        x = self.bn_3(x, training=training)
        x = self.conv_3(x)
        return x


class InceptionBackbone(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionBackbone, self).__init__(**kwargs)
        backbone = build_inception()
        self.backbone = tf.keras.Model(
            backbone.input,
            outputs={
                's1': backbone.get_layer('mixed2').output,
                's2': backbone.get_layer('mixed4').output,
                's3': backbone.get_layer('mixed7').output,
                's4': backbone.get_layer('mixed10').output,
            })

    def call(self, inputs, training=None):
        inputs = tf.keras.applications.inception_v3.preprocess_input(inputs)
        return self.backbone(inputs, training=training)


class GSCNN(tf.keras.Model):
    def __init__(self, n_classes, **kwargs):
        super(GSCNN, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = InceptionBackbone()
        self.shape_stream = ShapeStream()
        self.atrous_pooling = AtrousPyramidPooling(256)
        self.logit_layer = FinalLogitLayer(self.n_classes)

    def sobel_edges(self, tensor, eps=1e-8):
        gray = tf.image.rgb_to_grayscale(tensor[..., :3])
        tensor_edge = tf.image.sobel_edges(gray)
        mag = tf.reduce_sum(tensor_edge ** 2, axis=-1) + eps
        mag = tf.math.sqrt(mag)
        mag /= tf.reduce_max(mag, axis=[1, 2], keepdims=True)
        return mag

    def call(self, inputs, training=None, mask=None):
        input_shape = tf.shape(inputs)
        target_shape = tf.stack([input_shape[1], input_shape[2]])

        backbone_feature_dict = self.backbone(inputs, training=training)
        s1, s2, s3, s4 = (backbone_feature_dict['s1'],
                          backbone_feature_dict['s2'],
                          backbone_feature_dict['s3'],
                          backbone_feature_dict['s4'])
        backbone_features = [s1, s2, s3, s4]
        edge = self.sobel_edges(inputs)
        shape_activations, edge_out = self.shape_stream(
            [[backbone_features, edge], target_shape],
            training=training)
        backbone_activations = backbone_features[-1]
        intermediate_rep = backbone_features[1]
        net = self.atrous_pooling(
            [backbone_activations, shape_activations, intermediate_rep],
            training=training)
        net = self.logit_layer(net, training=training)
        net = tf.image.resize(net, target_shape)
        shape_activations = tf.image.resize(shape_activations, target_shape)
        return tf.concat([net, shape_activations], axis=-1)


if __name__ == '__main__':
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    g = GSCNN(10)
    # g(np.zeros([1, 200, 200, 3]))
    # g.summary(line_length=400)
