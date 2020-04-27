import tensorflow as tf

import cityscapes.dataset
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from gscnn.model_definition import GSCNN
from gscnn.train_and_evaluate import Trainer


batch_size = 8
network_input_h = network_input_w = 513
max_crop_downsample = 0.5
colour_aug_factor = 0.2
l1 = 1.
l2 = 10.
l3 = 1.
l4 = 1.
loss_weights = [l1, l2, l3, l4]
n_train_images = 2975
n_steps_in_epoch = n_train_images // batch_size

cityscapes_dataset_loader = cityscapes.dataset.CityScapes(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    data_dir='/media/ben/datasets/cityscapes')


model = GSCNN(n_classes=cityscapes.N_CLASSES)
# momentum = 0.9
# learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
#     7e-2,
#     n_steps_in_epoch * 230,
#     0.000001)
# optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)
optimiser = tf.keras.optimizers.RMSprop()
train_dataset = cityscapes_dataset_loader.build_training_dataset()
validation_dataset = cityscapes_dataset_loader.build_validation_dataset()

trainer = Trainer(
    model,
    train_dataset,
    validation_dataset,
    epochs=300,
    optimiser=optimiser,
    log_dir='logs',
    model_dir='logs/model',
    loss_weights=loss_weights,
    accumulation_iterations=None)
trainer.train_loop()









