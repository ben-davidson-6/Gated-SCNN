import os
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt


def list_files(startpath):
    """
    print structure of a directory
    :param startpath str directory to display contents of:
    :return:
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        if len(files) > 100:
            print('{}#{}files'.format(subindent, len(files)))
        else:
            for f in files:
                print('{}{}'.format(subindent, f))


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