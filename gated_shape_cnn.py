import tensorflow as tf
import layers


class InceptionBackbone(tf.keras.Model):
    def __init__(self, **kwargs):
        super(InceptionBackbone, self).__init__(**kwargs)
        base_net = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        base_net(tf.keras.layers.Input([None, None, 3]))
        self.backbone = tf.keras.Model(
            base_net.input,
            outputs={
                's1': base_net.get_layer('activation_2').output,
                's2': base_net.get_layer('mixed2').output,
                's3': base_net.get_layer('mixed7').output,
                's4': base_net.get_layer('mixed10').output,
            })
        self.backbone.trainable = False

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)


class GSCNN(tf.keras.Model):
    def __init__(self, n_classes, **kwargs):
        super(GSCNN, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = InceptionBackbone()
        self.shape_stream = None
        self.atrous_pooling = layers.AtrousPyramidPooling(256)
        self.logit_layer = None
        self.to_gray_scale = None
        self.resize = None

    def build(self, image_shape):
        self.shape_stream = layers.ShapeStream(image_shape[1], image_shape[2])
        self.logit_layer = layers.FinalLogitLayer(image_shape[1], image_shape[2], self.n_classes)
        self.resize = layers.Resize(image_shape[1], image_shape[2])
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

    # @tf.function
    def call(self, inputs, training=False):
        backbone_feature_dict = self.backbone(inputs)
        s1, s2, s3, s4 = backbone_feature_dict['s1'], backbone_feature_dict['s2'], backbone_feature_dict['s3'], backbone_feature_dict['s4']
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
        return net


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    gscnn = GSCNN(n_classes=2)
    gscnn.build((229, 229, 3))
    img = np.random.rand(1, 229, 229, 3)
    print(gscnn(img))
    # out, edge_out = gscnn(img, training=False)
    # oo = np.mean(edge_out, axis=-1)[0]
    # plt.imshow(oo)
    # plt.show()






