import tensorflow as tf

from gscnn.model_definition import GSCNN
from cityscapes.train_and_evaluate import Trainer
import cityscapes.dataset


batch_size = 6
network_input_h = network_input_w = 700
max_crop_downsample = 0.5
colour_aug_factor = 0.25
mixup_val = None
lr = 0.001
l1 = 1.
l2 = 10.
l3 = 1.
l4 = 1.

cityscapes_dataset_loader = cityscapes.dataset.CityScapes(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    data_dir='/home/ben/datasets/cityscapes',
    mixup_val=mixup_val,
)

model = GSCNN(n_classes=cityscapes.N_CLASSES)

n_train_images = 2975
n_steps_in_epoch = n_train_images // batch_size
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    0.001,
    n_steps_in_epoch*200,
    0.000001)
optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.5)

trainer = Trainer(
    model,
    cityscapes_dataset_loader.build_training_dataset(),
    cityscapes_dataset_loader.build_validation_dataset(),
    epochs=1000,
    optimiser=optimiser,
    log_dir='logs',
    model_dir='logs/model',
    l1=l1, l2=l2, l3=l3, l4=l4)
trainer.train_loop()





