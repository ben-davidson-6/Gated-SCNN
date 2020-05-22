import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

import gated_shape_cnn.model.model_definition

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from gated_shape_cnn.datasets.cityscapes.raw_dataset import CityScapesRaw
from gated_shape_cnn.datasets import cityscapes


def export(weights_path, out_p):
    """builds a saved model using weights at weights path"""
    gated_shape_cnn.model.model_definition.export_model(classes=cityscapes.N_CLASSES, ckpt_path=weights_path, out_dir=out_p, channels=3)


def show_single_example(model_dir):
    """plot image, edge activations, and segmentation of random image"""

    # get random example
    data = CityScapesRaw(cityscapes.DATA_DIR)
    img, label = data.get_random_val_example()
    label = np.where(label==255, 0, label)
    colour_label = cityscapes.TRAINING_COLOUR_PALETTE[label]

    # predict it!
    model = gated_shape_cnn.model.model_definition.GSCNNInfer(model_dir)
    pred, shape = model(img)
    pred = np.argmax(pred, axis=-1)
    colour_pred = cityscapes.TRAINING_COLOUR_PALETTE[pred]

    # plot it
    plt.subplot(3, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(3, 1, 2)
    plt.imshow(shape[0, ..., 0])
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(np.hstack((colour_label, colour_pred[0])))
    plt.axis('off')
    plt.show()


def build_results(model_dir):
    """
    Save the output segmentations for all validation images
    into CITYSCAPES_DATA_DIR/results. Then run the cityscapes
    evaluation scripts
    :param model_dir:
    :return:
    """
    lookup_arr = np.zeros([19], dtype=np.uint8)
    for i in range(19):
        lookup_arr[i] = cityscapes.TRAINID_TO_LABEL_ID[i]
    model = gated_shape_cnn.model.model_definition.GSCNNInfer(model_dir)
    data = CityScapesRaw(cityscapes.DATA_DIR)
    paths = data.get_img_paths(split=cityscapes.VAL)
    n = len(paths)

    for k, im_p in enumerate(paths):
        if k%10 == 0:
            print('done {} of {}'.format(k, n))
        name = os.path.basename(im_p)
        save_path = os.path.join(cityscapes.RESULTS_DIR, name)
        img = imageio.imread(im_p)
        pred, shape = model(img)
        pred = np.argmax(pred[0], axis=-1)
        pred = lookup_arr[pred].astype(np.uint8)
        imageio.imsave(save_path, pred)

    os.environ['CITYSCAPES_DATASET'] = cityscapes.DATA_DIR
    os.environ['CITYSCAPES_RESULTS'] = cityscapes.RESULTS_DIR
    import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling
    cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling.main()


def build_video_results(model_dir):
    model = gated_shape_cnn.model.model_definition.GSCNNInfer(model_dir)
    video_dir = '/home/ben/projects/gated_shape_cnns/stuttgart_00'
    video_results_dir = '/home/ben/projects/gated_shape_cnns/stuttgart_00_label'
    n = len(os.listdir(video_dir))
    for k, im_name in enumerate(os.listdir(video_dir)):
        if k%10 == 0:
            print(k, n)
        im_p = os.path.join(video_dir, im_name)
        target_p = os.path.join(video_results_dir, im_name)
        img = imageio.imread(im_p)
        pred, shape = model(img)
        pred = np.argmax(pred[0], axis=-1)
        pred = cityscapes.TRAINING_COLOUR_PALETTE[pred]

        result = img*0.5 + pred*0.5
        result = np.clip(result, 0, 255).astype(np.uint8)
        shape = (shape[0]*255).astype(np.uint8)
        result = np.vstack((result, np.tile(shape, (1, 1, 3))))
        imageio.imsave(target_p[:-3] + 'jpg', result)


if __name__ == '__main__':
    out_p = '/home/ben/projects/gated_shape_cnns/bestModel'
    # weights_path = '/home/ben/projects/gated_shape_cnns/logs/model/best'
    # export(weights_path, out_p)
    # show_single_example(out_p)
    # build_results(out_p)
    build_video_results(out_p)