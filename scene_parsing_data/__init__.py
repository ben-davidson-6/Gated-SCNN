import os
import pickle
import numpy as np

DATA_DIR = '/home/ben/datasets/ADEChallengeData2016'
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
TRAINING_IM_DIR = os.path.join(IMAGE_DIR, 'training')
VALIDATION_IM_DIR = os.path.join(IMAGE_DIR, 'validation')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
TRAINING_ANNOTATION_DIR = os.path.join(ANNOTATION_DIR, 'training')
VALIDATION_ANNOTATION_DIR = os.path.join(ANNOTATION_DIR, 'validation')
OBJECT_INFO_PATH = os.path.join(DATA_DIR, 'object_info.pickle')
COLORMAP_PATH = os.path.join(DATA_DIR, 'color150.npy')
N_CLASSES = 151

IMAGES = 'images'
LABELS = 'labels'
TRAINING_DIRS = {IMAGES: TRAINING_IM_DIR, LABELS: TRAINING_ANNOTATION_DIR}
VALIDATION_DIRS = {IMAGES: VALIDATION_IM_DIR, LABELS: VALIDATION_ANNOTATION_DIR}

COLOURS = np.load(COLORMAP_PATH)
BACKGROUND_COLOUR = np.zeros([1, 3], dtype=np.uint8)
COLOURS = np.concatenate([BACKGROUND_COLOUR, COLOURS], axis=0)

# dictionary with keys
# names, ratio, train, val
with open(OBJECT_INFO_PATH, 'rb') as pfile:
    OBJECT_INFO = pickle.load(pfile)
