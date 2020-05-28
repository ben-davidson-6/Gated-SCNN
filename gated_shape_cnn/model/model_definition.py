import imageio
import tensorflow as tf

from gated_shape_cnn.model.layers import (
    gradient_mag, ShapeStream, AtrousPyramidPooling, FinalLogitLayer, XceptionBackbone)


class GSCNN(tf.keras.Model):
    def __init__(self, n_classes, **kwargs):
        super(GSCNN, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = XceptionBackbone()
        self.shape_stream = ShapeStream()
        self.atrous_pooling = AtrousPyramidPooling(out_channels=256)
        self.logit_layer = FinalLogitLayer(self.n_classes)

    def call(self, inputs, training=None, mask=None):

        # we need to repeat the input if batch size is 1
        # because in training mode a batch size of 1 will create
        # nans, see:
        # https://github.com/tensorflow/tensorflow/issues/34062
        one_item_batch = tf.shape(inputs)[0] == 1
        if training is None:
            training = True
        inputs = tf.cond(
            tf.logical_and(one_item_batch, training),
            lambda: tf.tile(inputs, (2, 1, 1, 1)),
            lambda: inputs)

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
        edge = gradient_mag(inputs, from_rgb=True)
        shape_activations, edge_out = self.shape_stream(
            [backbone_features, edge],
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

        out = tf.cond(
            one_item_batch,
            lambda: out[:1],
            lambda: out)
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
        if type(p) == str:
            im = imageio.imread(p)
        else:
            im = p

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


if __name__ == '__main__':
    import numpy as np
    a = GSCNN(n_classes=2)
    a(np.random.random([1, 100, 100, 3]))