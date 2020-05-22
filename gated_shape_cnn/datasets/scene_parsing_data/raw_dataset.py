import os
import pickle
import imageio
import numpy as np
import multiprocessing
import sys

from scipy.io import loadmat

import gated_shape_cnn.datasets.utils
import gated_shape_cnn.training.utils
from gated_shape_cnn.datasets import scene_parsing_data


def matlab_mat_to_numpy():
    """read the colour palette and convert to an nd array"""
    if not os.path.exists(scene_parsing_data.COLORMAP_ORIG_PATH):
        os.system('wget {} -O {}'.format(scene_parsing_data.COLOR_DOWNLOAD_URL, scene_parsing_data.COLORMAP_ORIG_PATH))

    colors = loadmat(scene_parsing_data.COLORMAP_ORIG_PATH)['colors']
    background_colour = np.zeros([1, 3], dtype=np.uint8)
    colors = np.concatenate([background_colour, colors], axis=0)
    np.save(scene_parsing_data.COLORMAP_PATH[:-4], colors)


def parse_object_info():
    """Convert the text file to a a dictionary with properly typed values"""
    is_header = True
    # will contain integer_id -> info
    meta_data = {}
    with open(scene_parsing_data.ORIG_OBJECT_INFO_PATH, 'r') as text_file:
        for row in text_file:
            if is_header:
                is_header = False
                continue
            else:
                info = row.split()
                id_ = int(info[0])
                ratio = float(info[1])
                train = int(info[2])
                val = int(info[3])
                names = info[4]
                meta_data[id_] = {
                    'ratio': ratio,
                    'train': train,
                    'val': val,
                    'names': names,}
    with open(scene_parsing_data.OBJECT_INFO_PATH, 'wb') as pfile:
        pickle.dump(meta_data, pfile)


##################################################################
# Building the edge maps
#################################################################


def edge_path_from_label_path(label_path):
    label_name = os.path.basename(label_path)
    label_dir = os.path.dirname(label_path)
    edge_name = scene_parsing_data.EDGE_PREFIX + label_name
    edge_path = os.path.join(label_dir, edge_name)
    return edge_path


def label_path_to_edge_saved(label_path):
    edge_path = edge_path_from_label_path(label_path)
    label = imageio.imread(label_path)
    edge = gated_shape_cnn.training.utils.flat_label_to_edge_label(label, scene_parsing_data.N_CLASSES)
    imageio.imsave(edge_path, edge)


def create_edge_labels():
    pool = multiprocessing.Pool(16)
    train_labels = [os.path.join(scene_parsing_data.TRAINING_ANNOTATION_DIR, x) for x in os.listdir(
        scene_parsing_data.TRAINING_ANNOTATION_DIR)]
    val_labels = [os.path.join(scene_parsing_data.VALIDATION_ANNOTATION_DIR, x) for x in os.listdir(
        scene_parsing_data.VALIDATION_ANNOTATION_DIR)]

    num_train = len(train_labels)
    print('creating training edge maps')
    for i, _ in enumerate(pool.imap_unordered(label_path_to_edge_saved, train_labels), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / num_train))

    num_val = len(val_labels)
    print('creating val edge maps')
    for i, _ in enumerate(pool.imap_unordered(label_path_to_edge_saved, val_labels), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / num_val))


def get_dataset():
    if scene_parsing_data.DATA_DOWNLOAD_DIR is None:
        raise NotImplementedError('please specify the dataset directory in scene_parsing_data.__init__.py')

    # download the data and convert
    # the colour palette to numpy
    print('downloading raw scene parsing dataset and converting some')
    print('matlab files for use in python')
    os.system('wget {} -O {}'.format(scene_parsing_data.DATASET_URL, scene_parsing_data.DATA_DOWNLOAD_ZIP_PATH))
    os.system('unzip {} -d {}'.format(scene_parsing_data.DATA_DOWNLOAD_ZIP_PATH, scene_parsing_data.DATA_DOWNLOAD_DIR))
    os.remove(scene_parsing_data.DATA_DOWNLOAD_ZIP_PATH)
    print('converting object info txt file to python dictionary pickle')
    parse_object_info()
    print('downloading colour palette and converting to numpy array')
    matlab_mat_to_numpy()

    # build edge mask
    print('creating edge maps takes a long time!')
    create_edge_labels()
    print('FINIISHED!')
    print('your dataset directory looks like')
    gated_shape_cnn.datasets.utils.list_files(scene_parsing_data.DATA_DIR)


if __name__ == '__main__':
    pass
