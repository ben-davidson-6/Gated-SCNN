import os
import tensorflow as tf

import gated_shape_cnn.datasets.cityscapes
import gated_shape_cnn.datasets.cityscapes.dataset

from gated_shape_cnn.training.train_and_evaluate import train_model


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Build dataset
# define dataset parameters
batch_size = 4
network_input_h = network_input_w = 700
max_crop_downsample = 0.5
colour_aug_factor = 0.2

# build the dataset loader
data_dir_with_edge_maps = '/media/ben/datasets/cityscapes'
cityscapes_dataset_loader = gated_shape_cnn.datasets.cityscapes.dataset.CityScapes(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    debug=False,
    data_dir=data_dir_with_edge_maps)

# optimiser
n_train_images = 2975
n_steps_in_epoch = n_train_images // batch_size
momentum = 0.9
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    1e-2,
    n_steps_in_epoch * 230,
    0.000001)
optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)

# train
train_model(
    n_classes=cityscapes_dataset_loader.n_classes,
    train_data=cityscapes_dataset_loader.build_training_dataset(),
    val_data=cityscapes_dataset_loader.build_validation_dataset(),
    optimiser=optimiser,
    epochs=300,
    log_dir='./logs',
    model_dir='./logs/model',
    accum_iterations=4,
    loss_weights=(1., 1., 1., 1.)
)













