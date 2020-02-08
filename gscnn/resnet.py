import tensorflow.keras.layers as layers

from keras_applications.resnet_common import ResNet
from tensorflow.python.keras.applications import keras_modules_injection
from tensorflow.python.util.tf_export import keras_export

from gscnn.sync_norm import SyncBatchNormalization


@keras_export('keras.applications.resnet_v2.ResNet50V2',
              'keras.applications.ResNet50V2')
@keras_modules_injection
def Resnet50(*args, **kwargs):
    return ResNet50V2(*args, **kwargs)


def block2(x, filters, kernel_size=3, stride=1, dilate=False,
           conv_shortcut=False, name=None):
    """A residual block.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.

    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3

    preact = SyncBatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(
            4 * filters, 1,
            name=name + '_0_conv')(preact)
    else:
        if not dilate:
            shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x
        else:
            shortcut = x

    x = layers.Conv2D(
        filters,
        1,
        strides=1,
        use_bias=False,
        name=name + '_1_conv')(preact)
    x = SyncBatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    if dilate:
        x = layers.Conv2D(
            filters,
            kernel_size,
            dilation_rate=2,
            padding='SAME',
            use_bias=False,
            name=name + '_2_conv')(x)
    else:
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            name=name + '_2_conv')(x)
    x = SyncBatchNormalization(
        axis=bn_axis,
        epsilon=1.001e-5,
        name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack(x, filters, blocks, dilate, stride1=2, name=None):
    """A set of stacked residual blocks.

    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.

    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    if dilate:
        x = block2(x, filters, dilate=True, name=name + '_block' + str(blocks))
    else:
        x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))

    return x


def ResNet50V2(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               **kwargs):
    def stack_fn(x):
        x = stack(x, 64, 3, dilate=False, name='conv2')
        x = stack(x, 128, 4, dilate=False, name='conv3')
        x = stack(x, 256, 6, dilate=True, name='conv4')
        x = stack(x, 512, 3, dilate=False, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, True, True, 'resnet50v2',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)

