import os
import tensorflow as tf
import tensorflow.keras.layers
import gscnn.sync_norm


def modify_layers(model):
    # modify the network
    # convolutions to atrous
    convs_to_dilate = ['conv2d_26', 'conv2d_29', 'conv2d_71', 'conv2d_75']
    for layer_name in convs_to_dilate:
        # is set to (2, 2) in original model
        model.get_layer(layer_name).strides = (1, 1)

        # make atrous
        model.get_layer(layer_name).dilation_rate = 2
        model.get_layer(layer_name).padding = 'SAME'

    # maxpools cannot reduce now
    maxpools = ['max_pooling2d_2', 'max_pooling2d_3']
    for layer_name in maxpools:
        # is set to (2, 2) in original model
        model.get_layer(layer_name).strides = (1, 1)
        model.get_layer(layer_name).padding = 'SAME'


def build_inception():
    # monkey patch batch norm to synchronised and save to revert
    batch_norm = tensorflow.keras.layers.BatchNormalization
    tensorflow.keras.layers.BatchNormalization = gscnn.sync_norm.SyncBatchNormalization

    # build original model, save weights, we will modify the layers
    # so that the dilation rate of various convolutions is larger
    # creating atrous convolutions. We will also need to remove the downsampling layers
    model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=[None, None, 3])
    modify_layers(model)

    # rebuild new inception
    atrous_inception = tf.keras.models.model_from_json(model.to_json())
    atrous_inception.set_weights(model.get_weights())

    # reset batch norm
    tensorflow.keras.layers.BatchNormalization = batch_norm

    return atrous_inception



if __name__ == '__main__':
    build_inception()