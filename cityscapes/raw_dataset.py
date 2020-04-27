import os
import cityscapes
import os
import imageio
import numpy as np
import multiprocessing
from scipy.ndimage.morphology import distance_transform_edt
import sys
import glob
import random


class CityScapesRaw:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'leftImg8bit')
        assert os.path.exists(self.img_dir)
        self.label_dir = os.path.join(self.data_dir, 'gtFine')
        assert os.path.exists(self.label_dir)

    #####################################################
    # getting the correct paths relative to datadir
    #####################################################

    def get_image_split_dir(self, split):
        return os.path.join(self.img_dir, split)

    def get_label_split_dir(self, split):
        return os.path.join(self.label_dir, split)

    def get_img_paths(self, split):
        img_dir = self.get_image_split_dir(split)
        paths = glob.glob(img_dir + '/**/*.png', recursive=True)
        return paths

    @staticmethod
    def get_base_name_from_path(p):
        name = os.path.basename(p)
        base_name = '_'.join(name.split('_')[:3])
        return base_name

    @staticmethod
    def get_city_and_split(p):
        info = p.split(os.path.sep)
        # fname = info[-1]
        city = info[-2]
        split = info[-3]
        return city, split

    @staticmethod
    def get_meta_data_from_path(p):
        base_name = CityScapesRaw.get_base_name_from_path(p)
        city, split = CityScapesRaw.get_city_and_split(p)
        return base_name, city, split

    @staticmethod
    def build_all_names_from_base(base_name):
        img_name = base_name + cityscapes.IMG_SUFFIX
        label_name = base_name + cityscapes.LABEL_SUFFIX
        edge_label_name = base_name + cityscapes.EDGE_LABEL_SUFFIX
        return img_name, label_name, edge_label_name

    def build_image_dir(self, city, split):
        return os.path.join(self.get_image_split_dir(split), city)

    def build_label_dir(self, city, split):
        return os.path.join(self.get_label_split_dir(split), city)

    def convert_item_path_to_training_paths(self, p):
        base_name, city, split = CityScapesRaw.get_meta_data_from_path(p)
        img_name, label_name, edge_label_name = CityScapesRaw.build_all_names_from_base(base_name)

        img_dir = self.build_image_dir(city, split)
        label_dir = self.build_label_dir(city, split)

        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, label_name)
        edge_label_path = os.path.join(label_dir, edge_label_name)
        return img_path, label_path, edge_label_path

    def dataset_paths(self, split):
        img_paths = self.get_img_paths(split)
        dataset = [self.convert_item_path_to_training_paths(p) for p in img_paths]
        return dataset

    ####################################################################
    # create edge labels in the label directory of the cityscapes data
    ####################################################################

    @staticmethod
    def flat_label_to_edge_label(label, ):
        """
        Converts a segmentation label (H,W) to a binary edgemap (H,W)
        """
        radius = 2

        label[label == 255] = cityscapes.N_CLASSES
        one_hot_basis = np.eye(cityscapes.N_CLASSES + 1)
        one_hot = one_hot_basis[label]

        one_hot_pad = np.pad(one_hot, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        edgemap = np.zeros(one_hot.shape[:-1])

        for i in range(cityscapes.N_CLASSES + 1):
            dist = distance_transform_edt(one_hot_pad[..., i]) + \
                   distance_transform_edt(1.0 - one_hot_pad[..., i])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=-1)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap

    def create_edge_map_from_path(self, path):
        _, label_path, edge_path = self.convert_item_path_to_training_paths(path)
        label = imageio.imread(label_path)
        edge_label = CityScapesRaw.flat_label_to_edge_label(label)
        imageio.imsave(edge_path, edge_label)

    def build_edge_paths(self):
        p = multiprocessing.Pool(8)
        image_paths = self.get_img_paths(split=cityscapes.TRAIN)
        image_paths += self.get_img_paths(split=cityscapes.VAL)

        num_ps = len(image_paths)
        print('creating edge maps')
        for i, _ in enumerate(p.imap_unordered(self.create_edge_map_from_path, image_paths), 1):
            sys.stderr.write('\rdone {0:%}'.format(i / num_ps))

    ##########################################################
    # visualise
    ##########################################################

    def get_random_val_example(self):
        path = random.choice(self.get_img_paths(split=cityscapes.VAL))
        img_path, label_path, edge_path = self.convert_item_path_to_training_paths(path)
        img = imageio.imread(img_path)
        label = imageio.imread(label_path)
        return img, label

    def get_random_plottable_example(self):
        img, label = self.get_random_val_example()
        edge_label = CityScapesRaw.flat_label_to_edge_label(label)
        return img, label, edge_label

    def plot_random_val(self):
        img, label, edge_label = self.get_random_plottable_example()
        plt.subplot(3, 1, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(3, 1, 2)
        plt.imshow(label)
        plt.axis('off')
        plt.subplot(3, 1, 3)
        plt.imshow(edge_label[..., 0])
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    c = CityScapesRaw(cityscapes.DATA_DIR)
    c.build_edge_paths()
