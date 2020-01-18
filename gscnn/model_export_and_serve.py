import gscnn.model_definition as model_builder
import tensorflow as tf
import imageio


def export_model(h, w, c, classes, ckpt_path, out_dir):
    model = model_builder.GSCNN(classes)
    input = tf.keras.Input([h, w, c], dtype=tf.uint8)
    float_input = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(input)
    model(float_input, training=False)
    model.load_weights(ckpt_path)
    model.trainable = False

    output = model(float_input, training=False)
    o = tf.keras.layers.Lambda(tf.nn.softmax)(output[..., :-1])
    m = tf.keras.Model(input, [o, output[..., -1:]])
    m.trainable = False
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


if __name__ == '__main__':
    import cityscapes
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    p = '/home/ben/projects/gated_shape_cnns/logs/model/best.index'
    out = '/home/ben/projects/gated_shape_cnns/final_models/'
    export_model(1024, 2048, 3, cityscapes.N_CLASSES, p, out)

