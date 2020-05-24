import os
import imageio
import multiprocessing
import sys
import glob
import random

from gated_shape_cnn.training.utils import flat_label_to_edge_label
from gated_shape_cnn.datasets import cityscapes


class CityScapesRaw:
    """
    Process the CityScapes dataset under data_dir, process it to
    produce edge segmentations, and provide a self.dataset_paths() method
    for accessing the processed paths inside of the actual tf.Dataset class.

    Should only have to call build_edge_segs, and use the dataset_paths() inside
    of the dataset class
    """
    def __init__(self, data_dir):
        """
        :param data_dir str where your cityscapes data lives:
        """
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'leftImg8bit')
        assert os.path.exists(self.img_dir)
        self.label_dir = os.path.join(self.data_dir, 'gtFine')
        assert os.path.exists(self.label_dir)

    #####################################################
    # getting the correct paths relative to datadir
    #####################################################

    def _get_image_split_dir(self, split):
        return os.path.join(self.img_dir, split)

    def _get_label_split_dir(self, split):
        return os.path.join(self.label_dir, split)

    def get_img_paths(self, split):
        img_dir = self._get_image_split_dir(split)
        paths = glob.glob(img_dir + '/**/*.png', recursive=True)
        return paths

    @staticmethod
    def _get_base_name_from_path(p):
        name = os.path.basename(p)
        base_name = '_'.join(name.split('_')[:3])
        return base_name

    @staticmethod
    def _get_city_and_split(p):
        info = p.split(os.path.sep)
        # fname = info[-1]
        city = info[-2]
        split = info[-3]
        return city, split

    @staticmethod
    def _get_meta_data_from_path(p):
        base_name = CityScapesRaw._get_base_name_from_path(p)
        city, split = CityScapesRaw._get_city_and_split(p)
        return base_name, city, split

    @staticmethod
    def _build_all_names_from_base(base_name):
        img_name = base_name + cityscapes.IMG_SUFFIX
        label_name = base_name + cityscapes.LABEL_SUFFIX
        edge_label_name = base_name + cityscapes.EDGE_LABEL_SUFFIX
        return img_name, label_name, edge_label_name

    def _build_image_dir(self, city, split):
        return os.path.join(self._get_image_split_dir(split), city)

    def _build_label_dir(self, city, split):
        return os.path.join(self._get_label_split_dir(split), city)

    def _convert_item_path_to_training_paths(self, p):
        base_name, city, split = CityScapesRaw._get_meta_data_from_path(p)
        img_name, label_name, edge_label_name = CityScapesRaw._build_all_names_from_base(base_name)

        img_dir = self._build_image_dir(city, split)
        label_dir = self._build_label_dir(city, split)

        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, label_name)
        edge_label_path = os.path.join(label_dir, edge_label_name)
        return img_path, label_path, edge_label_path

    def dataset_paths(self, split):
        img_paths = self.get_img_paths(split)
        dataset = [self._convert_item_path_to_training_paths(p) for p in img_paths]
        return dataset

    ####################################################################
    # create edge labels in the label directory of the cityscapes data
    ####################################################################

    def _create_edge_map_from_path(self, path):
        _, label_path, edge_path = self._convert_item_path_to_training_paths(path)
        label = imageio.imread(label_path)
        edge_label = flat_label_to_edge_label(label, cityscapes.N_CLASSES)
        imageio.imsave(edge_path, edge_label)

    def build_edge_segs(self):
        p = multiprocessing.Pool(8)
        image_paths = self.get_img_paths(split=cityscapes.TRAIN)
        image_paths += self.get_img_paths(split=cityscapes.VAL)

        num_ps = len(image_paths)
        print('creating edge maps')
        for i, _ in enumerate(p.imap_unordered(self._create_edge_map_from_path, image_paths), 1):
            sys.stderr.write('\rdone {0:%}'.format(i / num_ps))

    ##########################################################
    # visualise
    ##########################################################

    def get_random_val_example(self):
        path = random.choice(self.get_img_paths(split=cityscapes.VAL))
        img_path, label_path, edge_path = self._convert_item_path_to_training_paths(path)
        img = imageio.imread(img_path)
        label = imageio.imread(label_path)
        return img, label

    def _get_random_plottable_example(self):
        img, label = self.get_random_val_example()
        edge_label = flat_label_to_edge_label(label, cityscapes.N_CLASSES)
        return img, label, edge_label

    def plot_random_val(self):
        img, label, edge_label = self._get_random_plottable_example()
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
    c.plot_random_val()
