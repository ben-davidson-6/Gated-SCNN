import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import cityscapes
from cityscapes.raw_dataset import CityScapesRaw
import gscnn.model_export_and_serve as inference
import matplotlib.pyplot as plt
import numpy as np
import imageio


out_p = '/home/ben/projects/gated_shape_cnns/final_models/1/'
weights_path = '/home/ben/venvs/py37/.guild/runs/09306376d4904d0995709ec0417d1cb1/logs/model/latest'


def export():
    inference.export_model(1024, 2048, 3, cityscapes.N_CLASSES, weights_path, out_p)


def show_single_example():
    data = CityScapesRaw(cityscapes.DATA_DIR)
    img, label = data.get_random_val_example()
    label = np.where(label==255, 0, label)
    colour_label = cityscapes.TRAINING_COLOUR_PALETTE[label]

    model = inference.GSCNNInfer(out_p)
    pred, shape = model(img)
    pred = np.argmax(pred, axis=-1)
    colour_pred = cityscapes.TRAINING_COLOUR_PALETTE[pred]

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


def build_results():
    lookup_arr = np.zeros([19], dtype=np.uint8)
    for i in range(19):
        lookup_arr[i] = cityscapes.TRAINID_TO_LABEL_ID[i]
    model = inference.GSCNNInfer(out_p)
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


if __name__ == '__main__':
    export()
    show_single_example()
    build_results()