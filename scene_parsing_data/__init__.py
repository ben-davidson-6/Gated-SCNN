import os
import pickle
import numpy as np


#################### SET ME ################################
DATA_DOWNLOAD_DIR = '/home/ben/datasets/'
#################### SET ME ################################


DATASET_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
DATA_DOWNLOAD_ZIP_PATH = os.path.join(DATA_DOWNLOAD_DIR, 'ADEChallengeData2016.zip')
DATA_DIR = DATA_DOWNLOAD_ZIP_PATH[:-4]
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
TRAINING_IM_DIR = os.path.join(IMAGE_DIR, 'training')
VALIDATION_IM_DIR = os.path.join(IMAGE_DIR, 'validation')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'annotations')
TRAINING_ANNOTATION_DIR = os.path.join(ANNOTATION_DIR, 'training')
VALIDATION_ANNOTATION_DIR = os.path.join(ANNOTATION_DIR, 'validation')
OBJECT_INFO_PATH = os.path.join(DATA_DIR, 'object_info.pickle')
ORIG_OBJECT_INFO_PATH = os.path.join(DATA_DIR, 'objectInfo150.txt')

COLOR_DOWNLOAD_URL = 'https://github.com/CSAILVision/sceneparsing/raw/master/visualizationCode/color150.mat'
COLORMAP_PATH = os.path.join(DATA_DIR, 'color150.npy')
COLORMAP_ORIG_PATH = os.path.join(DATA_DIR, 'color150.mat')
N_CLASSES = 151

IMAGES = 'images'
LABELS = 'labels'
TRAINING_DIRS = {IMAGES: TRAINING_IM_DIR, LABELS: TRAINING_ANNOTATION_DIR}
VALIDATION_DIRS = {IMAGES: VALIDATION_IM_DIR, LABELS: VALIDATION_ANNOTATION_DIR}

EDGE_PREFIX = 'edge_'
try:
    COLOURS = np.load(COLORMAP_PATH)
    # dictionary with keys
    # names, ratio, train, val
    with open(OBJECT_INFO_PATH, 'rb') as pfile:
        OBJECT_INFO = pickle.load(pfile)
except FileNotFoundError:
    COLOURS = None
    OBJECT_INFO = None




