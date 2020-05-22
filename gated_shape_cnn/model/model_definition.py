import imageio
import tensorflow as tf

from gated_shape_cnn.model.layers import (
    ShapeStream, AtrousPyramidPooling, FinalLogitLayer, XceptionBackbone)


class GSCNN(tf.keras.Model):
    def __init__(self, n_classes, **kwargs):
        super(GSCNN, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = XceptionBackbone()
        self.shape_stream = ShapeStream()
        self.atrous_pooling = AtrousPyramidPooling(out_channels=256)
        self.logit_layer = FinalLogitLayer(self.n_classes)

    def sobel_edges(self, tensor, eps=1e-12):
        gray = tf.image.rgb_to_grayscale(tensor[..., :3])
        tensor_edge = tf.image.sobel_edges(gray)
        mag = tf.reduce_sum(tensor_edge ** 2, axis=-1) + eps
        mag = tf.math.sqrt(mag)
        mag /= tf.reduce_max(mag, axis=[1, 2], keepdims=True)
        return mag

    def call(self, inputs, training=None, mask=None):
        # Backbone
        input_shape = tf.shape(inputs)
        target_shape = tf.stack([input_shape[1], input_shape[2]])
        backbone_feature_dict = self.backbone(inputs, training=training)
        s1, s2, s3, s4 = (backbone_feature_dict['s1'],
                          backbone_feature_dict['s2'],
                          backbone_feature_dict['s3'],
                          backbone_feature_dict['s4'])
        backbone_features = [s1, s2, s3, s4]

        # edge stream
        edge = self.sobel_edges(inputs)
        shape_activations, edge_out = self.shape_stream(
            [[backbone_features, edge], target_shape],
            training=training)

        # aspp
        backbone_activations = backbone_features[-1]
        intermediate_rep = backbone_features[1]
        net = self.atrous_pooling(
            [backbone_activations, shape_activations, intermediate_rep],
            training=training)

        # classify pixels
        net = self.logit_layer(net, training=training)
        net = tf.image.resize(net, target_shape)
        shape_activations = tf.image.resize(shape_activations, target_shape)
        out = tf.concat([net, shape_activations], axis=-1)

        return out


def export_model(classes, ckpt_path, out_dir, channels=3):
    """
    :param c channels:
    :param classes:
    :param ckpt_path:
    :param out_dir:
    :return:
    """

    # build the model and load the weights
    model = GSCNN(classes)
    input = tf.keras.Input([None, None, channels], dtype=tf.uint8)
    float_input = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(input)
    model(float_input, training=False)
    model.load_weights(ckpt_path)
    model.trainable = False

    # build the output with softmax so we get actual
    # predictions
    output = model(float_input, training=False)
    o = tf.keras.layers.Lambda(tf.nn.softmax)(output[..., :-1])
    m = tf.keras.Model(input, [o, output[..., -1:]])
    m.trainable = False

    # create saved model
    tf.saved_model.save(m, out_dir)


class GSCNNInfer:
    def __init__(self, saved_model_dir, resize=None):
        self.model = tf.saved_model.load(saved_model_dir)
        self.resize = resize

    def path_to_input(self, p):
        im = imageio.imread(p)
        if self.resize is not None:
            im = tf.image.resize(im, self.resize)
        return im

    def image_to_input(self, im):
        if len(im.shape) == 3:
            im = tf.expand_dims(im, axis=0)
        if self.resize is not None:
            im = tf.image.resize(im, self.resize)
        return im

    def __call__(self, im):
        im = self.image_to_input(im)
        class_pred, shape_head = self.model(im, training=False)
        return class_pred.numpy(), shape_head.numpy()