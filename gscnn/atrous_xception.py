import tensorflow as tf
import tensorflow.python.keras.applications.inception_v3


def modify_layers(model):

    # modify the network
    # convolutions to atrous
    convs_to_dilate = ['conv2d_3']
    rates = [2]
    for k, layer_name in enumerate(convs_to_dilate):
        # is set to (2, 2) in original model
        model.get_layer(layer_name).strides = (1, 1)

        # make atrous
        model.get_layer(layer_name).dilation_rate = rates[k]
        model.get_layer(layer_name).padding = 'SAME'

    # maxpools cannot reduce now
    maxpools = ['block13_pool']
    for layer_name in maxpools:
        # is set to (2, 2) in original model
        model.get_layer(layer_name).pool_size = (1, 1)
        model.get_layer(layer_name).strides = (1, 1)
        model.get_layer(layer_name).padding = 'SAME'


def build_xception():

    # build original model, save weights, we will modify the layers
    # so that the dilation rate of various convolutions is larger
    # creating atrous convolutions. We will also need to remove the downsampling layers
    model = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=[None, None, 3],)
    modify_layers(model)

    atrous_xception = tf.keras.models.model_from_json(model.to_json())
    atrous_xception.set_weights(model.get_weights())

    return atrous_xception


class AtrousXception(tf.keras.models.Model):
    def __init__(self, **kwargs):
        inception = build_xception()
        super(AtrousXception, self).__init__(inputs=inception.inputs, outputs=inception.outputs, **kwargs)


if __name__ == '__main__':
    import os
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    build_xception()
    #20,861,480