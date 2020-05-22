import tensorflow as tf

import gated_shape_cnn.datasets.cityscapes
import gated_shape_cnn.datasets.cityscapes.dataset
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"

from gated_shape_cnn.model import GSCNN
from gated_shape_cnn import GSCNNLoss
from gated_shape_cnn import SegmentationMeanIOU


# define dataset parameters
batch_size = 4
val_batch_size = 8
network_input_h = network_input_w = 800
max_crop_downsample = 0.5
colour_aug_factor = 0.2

# build the dataset
data_dir_with_edge_maps = '/media/ben/datasets/cityscapes'
cityscapes_dataset_loader = gated_shape_cnn.datasets.cityscapes.dataset.CityScapes(
    data_dir_with_edge_maps,
    batch_size,
    val_batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    build_for_keras=True,
    debug=False)

# build the loss
l1 = 1.
l2 = 20.
l3 = 1.
l4 = 1.
loss_weights = [l1, l2, l3, l4]
loss = GSCNNLoss(loss_weights)

# optimiser
n_train_images = 2975
n_steps_in_epoch = n_train_images // batch_size
momentum = 0.9
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    1e-2,
    n_steps_in_epoch * 230,
    0.000001)
optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)

# build the model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = GSCNN(n_classes=gated_shape_cnn.datasets.cityscapes.N_CLASSES)
    seg_metric = SegmentationMeanIOU(gated_shape_cnn.datasets.cityscapes.N_CLASSES)
    model.compile(
        optimizer=optimiser,
        loss=loss,
        metrics=seg_metric,
        )

model.fit(
    cityscapes_dataset_loader.build_training_dataset(),
    callbacks=[
        tf.keras.callbacks.TensorBoard('./cityscapesLogs'),
        tf.keras.callbacks.ModelCheckpoint(
            './cityscapesModel/model',
            monitor='val_segmentation_miou',
            save_best_only=True,
            mode='max')
    ],
    epochs=300,
    validation_data=cityscapes_dataset_loader.build_validation_dataset(),

)









