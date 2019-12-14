import scene_parsing_data
import os
import random
import imageio
import matplotlib.pyplot as plt
import numpy as np


def get_random_training_example_id():
    im_names = os.listdir(scene_parsing_data.TRAINING_IM_DIR)
    random_image_name = random.choice(im_names)
    return random_image_name[:-4]


def build_legend_info(object_ids):
    names = []
    colours = []
    for object_id in object_ids:
        if object_id == 0:
            names.append('other')
        else:
            object_info = scene_parsing_data.OBJECT_INFO[object_id]
            names.append(object_info['names'])
        colours.append(scene_parsing_data.COLOURS[object_id])
    return names, colours


def flat_label_to_plottable(label):
    coloured_image = scene_parsing_data.COLOURS[label]
    objects_present = np.unique(label)
    names, colours = build_legend_info(objects_present)
    return coloured_image, (names, colours)


def show_random_example():
    example_id = get_random_training_example_id()
    image_path = os.path.join(scene_parsing_data.TRAINING_IM_DIR, example_id + '.jpg')
    label_path = os.path.join(scene_parsing_data.TRAINING_ANNOTATION_DIR, example_id + '.png')
    image = imageio.imread(image_path)
    label = imageio.imread(label_path)
    coloured_label, (names, colours) = flat_label_to_plottable(label)
    plt.imshow(np.vstack((image, coloured_label)))
    plt.axis('off')
    plt.show()

show_random_example()