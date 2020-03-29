import tensorflow as tf

import cityscapes.dataset

from gscnn.model_definition import GSCNN
from gscnn.train_and_evaluate import Trainer


batch_size = 16
network_input_h = network_input_w = 650
max_crop_downsample = 0.4
colour_aug_factor = 0.1
mixup_val = None
l1 = 1.
l2 = 10.
l3 = 1.
l4 = 1.
n_train_images = 2975
n_steps_in_epoch = n_train_images // batch_size

cityscapes_dataset_loader = cityscapes.dataset.CityScapes(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    data_dir='/media/ben/datasets/cityscapes')


strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.OneDeviceStrategy('/gpu:0')


with strategy.scope():
    model = GSCNN(n_classes=cityscapes.N_CLASSES)
    momentum = 0.9
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        1e-2,
        n_steps_in_epoch * 230,
        0.000001)
    optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)

    train_dataset = strategy.experimental_distribute_dataset(
        cityscapes_dataset_loader.build_training_dataset())
    validation_dataset = strategy.experimental_distribute_dataset(
        cityscapes_dataset_loader.build_validation_dataset())

    trainer = Trainer(
        model,
        train_dataset,
        validation_dataset,
        epochs=300,
        optimiser=optimiser,
        log_dir='logs',
        model_dir='logs/model',
        l1=l1, l2=l2, l3=l3, l4=l4)
    trainer.train_loop()









