import tensorflow as tf

from gscnn.model_definition import GSCNN
from cityscapes.train_and_evaluate import Trainer
import cityscapes.dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

batch_size = 8
network_input_h = network_input_w = 800
max_crop_downsample = network_input_h/1024 * 0.90
colour_aug_factor = 0.25
mixup_val = None
lr = 0.001
l1 = 1.
l2 = 20.
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
    data_dir='/home/ben/datasets/cityscapes',)


strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])

with strategy.scope():
    model = GSCNN(n_classes=cityscapes.N_CLASSES)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        0.007,
        n_steps_in_epoch * 230,
        0.000001)
    optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
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









