import tensorflow as tf

import datasets.cityscapes
import datasets.cityscapes.dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from gscnn.model.model_definition import GSCNN
from gscnn.training.train_and_evaluate import Trainer


# define dataset parameters
batch_size = 4
network_input_h = network_input_w = 700
max_crop_downsample = 0.5
colour_aug_factor = 0.2

# loss hyperparams
l1 = 1.
l2 = 20.
l3 = 1.
l4 = 1.
loss_weights = [l1, l2, l3, l4]

# optimiser
n_train_images = 2975
n_steps_in_epoch = n_train_images // batch_size
momentum = 0.9
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    1e-2,
    n_steps_in_epoch * 230,
    0.000001)
optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)

# build the dataset loader
data_dir_with_edge_maps = '/media/ben/datasets/cityscapes'
cityscapes_dataset_loader = datasets.cityscapes.dataset.CityScapes(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    debug=False,
    data_dir=data_dir_with_edge_maps)

# build the model
model = GSCNN(n_classes=datasets.cityscapes.N_CLASSES)

# train
trainer = Trainer(
    model,
    cityscapes_dataset_loader.build_training_dataset(),
    cityscapes_dataset_loader.build_validation_dataset(),
    epochs=300,
    optimiser=optimiser,
    log_dir='logsRetrain',
    model_dir='logsRetrain/model',
    loss_weights=loss_weights,
    accumulation_iterations=4,)
trainer.train_loop()









