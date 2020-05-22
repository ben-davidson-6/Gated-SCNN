import tensorflow as tf

"""
    modify tf.keras.applications.Xception so that the last downsampling
    is replaced with atrous convolution, and we add some regularization
    as in the original xception paper
"""


def modify_layers(model):
    """
    take the tf.keras.Model and modify the parameters of the layers.
    We will then rebuild the model from json, which will have the updated layers
    but we should still be able to use the pretrained weights
    """

    # modify the last downsampling convolutions
    # convolutions to atrous
    convs_to_dilate = ['conv2d_3']
    rates = [2]
    for k, layer_name in enumerate(convs_to_dilate):
        # is set to (2, 2) in original model
        model.get_layer(layer_name).strides = (1, 1)

        # make atrous
        model.get_layer(layer_name).dilation_rate = rates[k]
        model.get_layer(layer_name).padding = 'SAME'

    # We also need to turn this maxpool into the identity
    # so that the shapes match up, there is no point
    # in running a max pool filter but keeping the same size
    maxpools = ['block13_pool']
    for layer_name in maxpools:
        # is set to (2, 2) in original model
        model.get_layer(layer_name).pool_size = (1, 1)
        model.get_layer(layer_name).strides = (1, 1)
        model.get_layer(layer_name).padding = 'SAME'

    # add some weight decay
    for layer in model.layers:
        model.get_layer(layer.name).kernel_regularizer = tf.keras.regularizers.l2(l=1e-5)


def build_xception():
    """
    Create an atrous version of tf.keras.applications.Xception
    which uses the pretrained image net weights
    """

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
    pass
