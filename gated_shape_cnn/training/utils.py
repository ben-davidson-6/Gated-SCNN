import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt


def validate_tensor_rank(tensor, name):
    rank = tf.rank(tensor)
    tf.assert_equal(
        rank,
        4,
        message='{} should be rank 4, saw: {}'.format(name, rank))


def validate_edge_tensor(edge):
    validate_tensor_rank(edge, 'edges')
    edge_channels = tf.shape(edge)[-1]
    edge_channel_str = 'edges should have 2 channels, saw: {}'.format(edge_channels)
    tf.assert_equal(
        edge_channels,
        2,
        message=edge_channel_str)


def validate_label_tensor(label):
    validate_tensor_rank(label, 'labels')
    label_channels = tf.shape(label)[-1]
    label_channel_str = 'labels should have at least 2 channels, saw {}'.format(label_channels)
    tf.assert_greater(
        label_channels,
        1,
        message=label_channel_str)


def validate_image_tensor(image):
    validate_tensor_rank(image, 'images')
    image_channels = tf.shape(image)[-1]
    image_channel_str = 'images should have at least 3 channels, saw {}'.format(image_channels)
    tf.assert_equal(
        image_channels,
        3,
        message=image_channel_str)


def _label_to_one_hot(label, n_classes):
    """
        Converts a segmentation mask (H,W) to (H,W,K) where the last dim is a one
        hot encoding vector
        """
    _mask = [label == (i + 1) for i in range(n_classes)]
    return np.stack(_mask, axis=-1).astype(np.uint8)


def flat_label_to_edge_label(label, n_classes, radius=2):
    """
    Converts a segmentation label (H,W) to a binary edgemap (H,W)
    """
    one_hot = _label_to_one_hot(label, n_classes)
    one_hot_pad = np.pad(one_hot, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    edgemap = np.zeros(one_hot.shape[:-1])

    for i in range(n_classes):
        dist = distance_transform_edt(one_hot_pad[..., i]) + \
               distance_transform_edt(1.0 - one_hot_pad[..., i])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=-1)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap