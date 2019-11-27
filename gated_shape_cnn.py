import tensorflow as tf
tf.keras.layers.BatchNormalization._USE_V2_BEHAVIOR = False


class Resize(tf.keras.layers.Layer):
    def __init__(self, h, w, **kwargs):
        super(Resize, self).__init__(**kwargs)
        self.target_shape = tf.stack([h, w])

    def call(self, inputs, **kwargs):
        return tf.image.resize(inputs, self.target_shape)


class GateConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GateConv, self).__init__(**kwargs)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.conv_1 = None
        self.relu = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv2D(1, kernel_size=1)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(in_channels, kernel_size=1)

    def call(self, x, training=False):
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

    def call(self, x, **kwargs):
        feature_map, shape_map = x
        features = tf.concat([feature_map, shape_map], axis=-1)
        alpha = self.gated_conv(features)
        gated = feature_map*(alpha + 1.)
        return self.conv_1(gated)


class ResnetPreactUnit(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(ResnetPreactUnit, self).__init__(**kwargs)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv_1 = tf.keras.layers.Conv2D(depth, 3, padding='SAME')
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(depth, 3, padding='SAME')

    def call(self, x, training=False):
        shortcut = x
        x = self.bn_1(x, training=training)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x, training=training)
        x = self.relu(x)
        x = self.conv_2(x)
        return x + shortcut


class ShapeAttention(tf.keras.layers.Layer):
    def __init__(self, h, w, **kwargs):
        super(ShapeAttention, self).__init__(**kwargs)
        self.resize = Resize(h, w)

        self.gated_conv_1 = GatedShapeConv()
        self.gated_conv_2 = GatedShapeConv()
        self.gated_conv_3 = GatedShapeConv()

        self.shape_reduction_1 = tf.keras.layers.Conv2D(1, 1)
        self.shape_reduction_2 = tf.keras.layers.Conv2D(1, 1)
        self.shape_reduction_3 = tf.keras.layers.Conv2D(1, 1)
        self.shape_reduction_4 = tf.keras.layers.Conv2D(1, 1)

        self.res_1 = ResnetPreactUnit(64)
        self.res_2 = ResnetPreactUnit(32)
        self.res_3 = ResnetPreactUnit(16)

        self.reduction_conv_1 = tf.keras.layers.Conv2D(32, 1)
        self.reduction_conv_2 = tf.keras.layers.Conv2D(16, 1)
        self.reduction_conv_3 = tf.keras.layers.Conv2D(8, 1)
        self.reduction_conv_4 = tf.keras.layers.Conv2D(1, 1, use_bias=False)
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, x, training=False):
        s1, s2, s3, s4 = x
        # todo this resizing can be made better
        s1 = self.shape_reduction_1(s1)
        s1 = self.resize(s1)
        s2 = self.shape_reduction_2(s2)
        s2 = self.resize(s2)
        s3 = self.shape_reduction_3(s3)
        s3 = self.resize(s3)
        s4 = self.shape_reduction_4(s4)
        s4 = self.resize(s4)

        x = self.res_1(s1, training=training)
        x = self.reduction_conv_1(x)
        x = self.gated_conv_1([x, s2], training=training)

        x = self.res_2(x, training=training)
        x = self.reduction_conv_2(x)
        x = self.gated_conv_2([x, s3], training=training)

        x = self.res_3(x, training=training)
        x = self.reduction_conv_3(x)
        x = self.gated_conv_3([x, s4], training=training)

        x = self.reduction_conv_4(x)
        x = self.sigmoid(x)

        return x


class ShapeStream(tf.keras.layers.Layer):
    def __init__(self, h, w, **kwargs):
        super(ShapeStream, self).__init__(**kwargs)
        self.shape_attention = ShapeAttention(h, w)
        self.reduction_conv = tf.keras.layers.Conv2D(2, 1, use_bias=False)
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def call(self, x, training=False):
        shape_backbone_activations, image_edges = x
        edge_out = self.shape_attention(shape_backbone_activations)
        backbone_representation = tf.concat([edge_out, image_edges], axis=-1)
        shape_logits = self.reduction_conv(backbone_representation)
        shape_attention = self.sigmoid(shape_logits)
        return shape_attention, edge_out


class AtrousConvolution(tf.keras.layers.Layer):
    def __init__(self, rate, filters, kernel_size, use_bias, activation, **kwargs):
        super(AtrousConvolution, self).__init__(**kwargs)
        self.pad = tf.keras.layers.ZeroPadding2D((rate, rate))
        self.convolution = tf.keras.layers.Conv2D(filters, kernel_size, activation=activation, use_bias=use_bias, dilation_rate=(rate, rate))

    def call(self, x, training=False):
        return self.convolution(self.pad(x))


class AtrousPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(AtrousPyramidPooling, self).__init__(**kwargs)

        # for final output of backbone
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(out_channels, 1, activation=tf.nn.relu)

        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.atrous_conv_1 = AtrousConvolution(6, filters=out_channels, kernel_size=3, use_bias=False, activation=tf.nn.relu)

        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.atrous_conv_2 = AtrousConvolution(12, filters=out_channels, kernel_size=3, use_bias=False, activation=tf.nn.relu)

        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.atrous_conv_3 = AtrousConvolution(18, filters=out_channels, kernel_size=3, use_bias=False, activation=tf.nn.relu)

        # for backbone features
        self.bn_img = tf.keras.layers.BatchNormalization()
        self.conv_img = tf.keras.layers.Conv2D(out_channels, 1, activation=tf.nn.relu)

        # for shape features
        self.bn_shape = tf.keras.layers.BatchNormalization()
        self.conv_shape = tf.keras.layers.Conv2D(out_channels, 1, activation=tf.nn.relu)

        # 1x1 reduction convolutions
        self.conv_reduction_1 = tf.keras.layers.Conv2D(64, 1, use_bias=False)
        self.conv_reduction_2 = tf.keras.layers.Conv2D(256, 1, use_bias=False)

        self.resize_backbone = None
        self.resize_intermediate = None

    def call(self, x, training=False):
        image_features, shape_features, intermediate_rep = x

        backbone_shape, intermediate_shape = tf.shape(image_features), tf.shape(intermediate_rep)
        self.resize_backbone = Resize(backbone_shape[1], backbone_shape[2])
        self.resize_intermediate = Resize(intermediate_shape[1], intermediate_shape[2])

        # process backbone features and the shape activations
        # from the shape stream
        img_net = tf.reduce_mean(image_features, axis=[1, 2], keepdims=True)
        img_net = self.bn_img(img_net, training=training)
        img_net = self.conv_img(img_net)
        img_net = self.resize_backbone(img_net)
        shape_net = self.resize_backbone(shape_features)
        shape_net = self.bn_shape(shape_net, training=training)
        net = tf.concat([img_net, shape_net], axis=-1)

        # process with atrous
        w = self.bn_1(image_features, training=training)
        w = self.conv_1(w)
        x = self.bn_2(image_features, training=training)
        x = self.atrous_conv_1(x)
        y = self.bn_3(image_features, training=training)
        y = self.atrous_conv_2(y)
        z = self.bn_4(image_features, training=training)
        z = self.atrous_conv_3(z)

        # atrous output from final layer of backbone
        # and shape stream
        net = tf.concat([net, w, x, y, z], axis=-1)
        net = self.conv_reduction_1(net)

        # combine intermediate representation
        intermediate_rep = self.conv_reduction_2(intermediate_rep)
        net = self.resize_intermediate(net)
        net = tf.concat([net, intermediate_rep], axis=-1)

        return net


class FinalLogitLayer(tf.keras.layers.Layer):
    def __init__(self, h, w, num_classes, **kwargs):
        super(FinalLogitLayer, self).__init__(**kwargs)
        self.resize = Resize(h, w)
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(256, 3, padding='SAME', use_bias=False, activation=tf.nn.relu)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(256, 3, padding='SAME', use_bias=False, activation=tf.nn.relu)
        self.conv_3 = tf.keras.layers.Conv2D(num_classes, 1, padding='SAME', use_bias=False)

    def call(self, x, training=False):
        x = self.bn_1(x, training=training)
        x = self.conv_1(x)
        x = self.bn_2(x, training=training)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.resize(x)
        return x


class InceptionBackbone(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InceptionBackbone, self).__init__(**kwargs)
        self.backbone = None

    def build(self, input_shape):
        if len(input_shape) == 4:
            input_shape = input_shape[1:]
        backbone = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=None,
            input_shape=input_shape)
        self.backbone = tf.keras.Model(
            backbone.input,
            outputs={
                's1': backbone.get_layer('activation_5').output,
                's2': backbone.get_layer('mixed2').output,
                's3': backbone.get_layer('mixed7').output,
                's4': backbone.get_layer('mixed10').output,
            })

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)


class GSCNN(tf.keras.Model):
    def __init__(self, n_classes, **kwargs):
        super(GSCNN, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = InceptionBackbone()
        self.shape_stream = None
        self.atrous_pooling = AtrousPyramidPooling(256)
        self.logit_layer = None
        self.to_gray_scale = None
        self.resize = None

    def build(self, image_shape):
        self.shape_stream = ShapeStream(image_shape[1], image_shape[2])
        self.logit_layer = FinalLogitLayer(image_shape[1], image_shape[2], self.n_classes)
        self.resize = Resize(image_shape[1], image_shape[2])
        ndim = len(image_shape)
        if ndim == 3:
            self.to_gray_scale = tf.image.rgb_to_grayscale
        elif ndim == 1:
            self.to_gray_scale = lambda x: x
        else:
            self.to_gray_scale = tf.keras.layers.Conv2D(1, 1, use_bias=False)

    def get_edge_image(self, tensor):
        gray = self.to_gray_scale(tensor)
        sobel = tf.image.sobel_edges(gray)
        mag = tf.linalg.norm(sobel, axis=-1)
        return mag/tf.reduce_max(mag)

    @tf.function
    def call(self, inputs, training=False):
        backbone_feature_dict = self.backbone(inputs, training)
        s1, s2, s3, s4 = (backbone_feature_dict['s1'],
                          backbone_feature_dict['s2'],
                          backbone_feature_dict['s3'],
                          backbone_feature_dict['s4'])
        backbone_features = [s1, s2, s3, s4]
        edge = self.get_edge_image(inputs)
        shape_activations, edge_out = self.shape_stream(
            [backbone_features, edge],
            training)
        backbone_activations = backbone_features[-1]
        intermediate_rep = backbone_features[1]
        net = self.atrous_pooling(
            [backbone_activations, shape_activations, intermediate_rep],
            training)
        net = self.logit_layer(net, training)
        net = self.resize(net)
        return net, shape_activations


class DebugModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DebugModel, self).__init__(**kwargs)
        self.d1 = tf.keras.layers.Conv2D(16, 3, activation=tf.nn.relu)
        self.d2 = tf.keras.layers.Conv2D(32, 3, activation=tf.nn.relu)
        self.d4 = tf.keras.layers.Conv2D(11, 3, activation=None, use_bias=False)

    @tf.function
    def call(self, inputs, training=False):
        s = tf.image.sobel_edges(inputs)
        s = tf.linalg.norm(s, axis=-1)
        s /= tf.reduce_max(s, axis=-1, keepdims=True)

        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d4(x)
        x = tf.image.resize(x, [28, 28])

        return x, s

if __name__ == '__main__':
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    gscnn = GSCNN(2)
    t = gscnn(np.random.random([1, 100, 100, 3]))
    print(t)