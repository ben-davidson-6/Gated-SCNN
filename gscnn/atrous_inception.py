import tensorflow as tf
import tensorflow.python.keras.applications.inception_v3


def conv2d_sync_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = 3
    x = tf.keras.layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = tf.keras.layers.experimental.SyncBatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = tf.keras.layers.Activation('relu', name=name)(x)
    return x


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

    # monkey patch keras
    original_conv2d_bn = tensorflow.python.keras.applications.inception_v3.conv2d_bn
    tensorflow.python.keras.applications.inception_v3.conv2d_bn = conv2d_sync_bn

    # build original model, save weights, we will modify the layers
    # so that the dilation rate of various convolutions is larger
    # creating atrous convolutions. We will also need to remove the downsampling layers
    model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=[None, None, 3])
    modify_layers(model)

    for layer in model.layers:
        layer.kernel_regularizer = tf.keras.regularizers.l2(l=1e-6)

    # rebuild new inception
    atrous_inception = tf.keras.models.model_from_json(model.to_json())
    atrous_inception.set_weights(model.get_weights())
    # atrous_inception.summary(line_length=300)

    # reset monkey patch
    tensorflow.python.keras.applications.inception_v3 = original_conv2d_bn
    return atrous_inception



if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    build_inception()