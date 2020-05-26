import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt


def validate_edge_tensor(edge):
    tf.debugging.assert_shapes(
        [(edge, ('b', 'h', 'w', 2))],
        message='edges')
    tf.debugging.assert_type(
        edge,
        tf.float32,
        message='edges')


def validate_label_tensor(label):
    tf.debugging.assert_rank(
       label,
        4,
        message='label')
    # raise value error for consistency with other validations
    try:
        tf.debugging.assert_greater_equal(
            tf.shape(label)[-1],
            2)
    except tf.errors.InvalidArgumentError:
        raise ValueError('must have at least 2 channels in label')
    tf.debugging.assert_type(
        label,
        tf.float32,
        message='label')


def validate_image_tensor(image):
    tf.debugging.assert_shapes(
        [(image, ('b', 'h', 'w', 3))],
        message='image'
    )
    tf.debugging.assert_type(
        image,
        tf.float32,
        message='image')


def _label_to_one_hot_for_boundary(label, n_classes):
    """
        Converts a segmentation mask (H,W) to (H,W,K) where the last dim is a one
        hot encoding vector
        """
    assert label.ndim == 2, 'label must be of shape (h, w)'
    _mask = []
    for i in range(n_classes):
        _mask.append(np.isclose(label, i))
    return np.stack(_mask, axis=-1).astype(np.uint8)


def flat_label_to_edge_label(label, n_classes, radius=2):
    """
    Converts a segmentation label (H,W) to a binary edgemap (H, W, 1)
    """
    one_hot = _label_to_one_hot_for_boundary(label, n_classes)
    one_hot_pad = np.pad(one_hot, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    edgemap = np.zeros(one_hot.shape[:-1])
    classless_pixels = np.all(one_hot == 0, axis=-1)
    for i in range(n_classes):
        dist_of = one_hot_pad[..., i] #+ classless_pixels.astype(np.float32)

        dist = distance_transform_edt(1.0 - dist_of)
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap[classless_pixels] = 0
    edgemap = np.expand_dims(edgemap, axis=-1)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap