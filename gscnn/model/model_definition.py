import tensorflow as tf
from gscnn.model.atrous_xception import AtrousXception
from gscnn.model.layers import ShapeStream, AtrousPyramidPooling, FinalLogitLayer, XceptionBackbone


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


if __name__ == '__main__':
    pass

