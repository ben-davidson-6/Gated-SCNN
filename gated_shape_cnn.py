import tensorflow as tf
import layers


class GSCNN(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = None
        self.shape_stream = None
        self.atrous_pooling = None
        self.logit_layer = None
        self.resize = None

    def build(self, input_shape):
        image_shape, _ = input_shape
        assert len(image_shape) == 4
        self.backbone = tf.keras.applications.InceptionV3(
            include_top=False,
            weights=None,
            input_shape=image_shape[1:])
        s1 = self.backbone.get_layer('activation_2').output
        s2 = self.backbone.get_layer('mixed2').output
        s3 = self.backbone.get_layer('mixed7').output
        s4 = self.backbone.get_layer('mixed10').output
        self.backbone = tf.keras.Model(self.backbone.input,  [s1, s2, s3, s4])
        self.shape_stream = layers.ShapeStream(image_shape[1], image_shape[2])
        self.atrous_pooling = layers.AtrousPyramidPooling(256)
        self.logit_layer = layers.FinalLogitLayer(image_shape[1], image_shape[2])
        self.resize = layers.Resize(image_shape[1], image_shape[2])

    def call(self, inputs, training=None, mask=None):
        image, edge = inputs
        backbone_features = self.backbone(image, training=training)

        shape_activations = self.shape_stream(
            [backbone_features, edge],
            training)
        backbone_activations = backbone_features[-1]
        intermediate_rep = backbone_features[1]

        net = self.atrous_pooling(
            [backbone_activations, shape_activations, intermediate_rep],
            training)
        net = self.logit_layer(net, training)
        net = self.resize(net)
        return net


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    gscnn = GSCNN()
    img = np.random.rand(1, 229, 229, 3)
    edge = np.random.rand(1, 229, 229, 1)
    out = gscnn([img, edge], training=False)
    plt.imshow(np.mean(out, axis=-1)[0])
    plt.show()






