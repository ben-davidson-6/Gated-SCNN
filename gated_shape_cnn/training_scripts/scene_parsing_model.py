import tensorflow as tf

from gated_shape_cnn import datasets
import gated_shape_cnn.datasets.scene_parsing_data.dataset
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from gated_shape_cnn import GSCNN
from gated_shape_cnn import Trainer


# define dataset parameters
batch_size = 8
network_input_h = network_input_w = 256
max_crop_downsample = 0.95
colour_aug_factor = 0.2

# loss hyperparams
l1 = 1.
l2 = 20.
l3 = 1.
l4 = 1.
loss_weights = [l1, l2, l3, l4]

# optimiser
n_train_images = 20210
n_steps_in_epoch = n_train_images // batch_size
momentum = 0.9
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    1e-2,
    n_steps_in_epoch * 230,
    0.000001)
optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)
# optimiser = tf.keras.optimizers.RMSprop()

# build the dataset loader
dataset_loader = gated_shape_cnn.datasets.scene_parsing_data.dataset.SceneParsing(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    build_for_keras=False,
    debug=False)

# build the model
model = GSCNN(n_classes=gated_shape_cnn.datasets.scene_parsing_data.N_CLASSES)

# train
trainer = Trainer(
    model,
    dataset_loader.build_training_dataset(),
    dataset_loader.build_validation_dataset(),
    epochs=300,
    optimiser=optimiser,
    log_dir='logsScene',
    model_dir='logsScene/model',
    loss_weights=loss_weights,
    accumulation_iterations=2,)
trainer.train_loop()









