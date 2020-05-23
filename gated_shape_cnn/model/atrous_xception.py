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

    Originally got this by name, but the name depends on what you call before hand.
    To get the right layer we get what index each are at using the snippet below
    and then hardcode these indices to the code
    # for k, layer in enumerate(model.layers):
    #     if layer.name == 'conv2d_3':
    #         print('conv', k)
    #     if layer.name == 'block13_pool':
    #         print('pool', k)
    #     if layer.name == 'add_6':
    #         print('add', k)
    """
    # modify the last downsampling convolutions
    # we cant get by name as this changes dependening
    # on what you do before!
    conv_layer_index = 122
    # is set to (2, 2) in original model
    model.layers[conv_layer_index].strides = (1, 1)
    # make atrous
    model.layers[conv_layer_index].dilation_rate = 2
    model.layers[conv_layer_index].padding = 'SAME'

    # We also need to turn this maxpool into the identity
    # so that the shapes match up, there is no point
    # in running a max pool filter but keeping the same size
    pool_layer_index = 123
    # maxpools = ['block13_pool']
    model.layers[pool_layer_index].pool_size = (1, 1)
    model.layers[pool_layer_index].strides = (1, 1)
    model.layers[pool_layer_index].padding = 'SAME'

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
    build_xception()
