import tensorflow as tf

from gscnn.model.atrous_xception import AtrousXception


def resize_to(x, target_t=None, target_shape=None):
    """resize x to shape or target_tensor or target_shape"""
    if target_shape is None:
        s = tf.shape(target_t)
        target_shape = tf.stack([s[1], s[2]])
    return tf.image.resize(x, target_shape, )


class GateConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GateConv, self).__init__(**kwargs)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(
            fused=False,
            scale=False,
            momentum=0.9)
        self.conv_1 = None
        self.relu = tf.keras.layers.ReLU()
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(
            fused=False,
            momentum=0.9)
        self.sigmoid = tf.keras.layers.Activation(tf.nn.sigmoid)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=in_channels,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

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
        self.conv_1 = tf.keras.layers.Conv2D(
            feature_channels,
            1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def call(self, x, training=None):
        feature_map, shape_map = x
        features = tf.concat([feature_map, shape_map], axis=-1)
        alpha = self.gated_conv(features, training=training)
        gated = feature_map*(alpha + 1.)
        return self.conv_1(gated)


class ResnetPreactUnit(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ResnetPreactUnit, self).__init__(**kwargs)
        self.bn_1 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.relu = tf.keras.layers.ReLU()
        self.conv_1 = None
        self.bn_2 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_2 = None
        self.add = tf.keras.layers.Add()

    def build(self, input_shape):
        cs = input_shape[-1]

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=cs,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=cs,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def call(self, x, training=None):
        shortcut = x
        x = self.bn_1(x, training)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_2(x, training)
        x = self.relu(x)
        x = self.conv_2(x)
        return self.add([x, shortcut])


class ShapeAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ShapeAttention, self).__init__(**kwargs)

        self.gated_conv_1 = GatedShapeConv()
        self.gated_conv_2 = GatedShapeConv()
        self.gated_conv_3 = GatedShapeConv()

        self.shape_reduction_2 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.shape_reduction_3 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.shape_reduction_4 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        self.res_1 = ResnetPreactUnit()
        self.res_2 = ResnetPreactUnit()
        self.res_3 = ResnetPreactUnit()

        self.reduction_conv_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.reduction_conv_2 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.reduction_conv_3 = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.reduction_conv_4 = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
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
        self.reduction_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
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
        self.depthwise_kernel = None
        self.pointwise_kernel = None
        self.channel_multiplier = 1

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.depthwise_kernel = self.add_weight(
            name='kernel',
            shape=[self.kernel_size, self.kernel_size, in_channels, self.channel_multiplier],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.pointwise_kernel = self.add_weight(
            name='kernel',
            shape=[1, 1, in_channels*self.channel_multiplier, self.out_channels],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4))

    def call(self, x, training=None):
        return tf.nn.separable_conv2d(
            x,
            self.depthwise_kernel,
            self.pointwise_kernel,
            strides=[1, 1, 1, 1],
            dilations=[self.rate, self.rate],
            padding='SAME', )


class AtrousPyramidPooling(tf.keras.layers.Layer):

    def __init__(self, out_channels, **kwargs):
        super(AtrousPyramidPooling, self).__init__(**kwargs)
        self.relu = tf.keras.layers.ReLU()

        # for final output of backbone
        self.bn_1 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        self.bn_2 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.atrous_conv_1 = AtrousConvolution(rate=6, filters=out_channels, kernel_size=3)

        self.bn_3 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.atrous_conv_2 = AtrousConvolution(rate=12, filters=out_channels, kernel_size=3)

        self.bn_4 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.atrous_conv_3 = AtrousConvolution(rate=18, filters=out_channels, kernel_size=3)

        # for backbone features
        self.bn_img = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_img = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        # for shape features
        self.bn_shape = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_shape = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

        # 1x1 reduction convolutions
        self.conv_reduction_1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.conv_reduction_2 = tf.keras.layers.Conv2D(
            filters=48,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

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
        self.bn_1 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.bn_2 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.conv_2 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding='SAME',
            use_bias=False,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))
        self.bn_3 = tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)

        self.conv_3 = tf.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding='SAME',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))

    def call(self, x, training=None):
        x = self.bn_1(x, training=training)
        x = self.conv_1(x)
        x = self.bn_2(x, training=training)
        x = self.conv_2(x)
        x = self.bn_3(x, training=training)
        x = self.conv_3(x)
        return x


class XceptionBackbone(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(XceptionBackbone, self).__init__(**kwargs)
        self.backbone = None
        backbone = AtrousXception()
        self.backbone = tf.keras.Model(
            backbone.input,
            outputs={
                's1': backbone.get_layer('block2_sepconv2_bn').output,
                's2': backbone.get_layer('block3_sepconv2_bn').output,
                's3': backbone.get_layer('add_6').output,
                's4': backbone.get_layer('block14_sepconv2_act').output,
            })

    def call(self, inputs, training=None):
        inputs = tf.keras.applications.xception.preprocess_input(inputs)
        return self.backbone(inputs, training=training)