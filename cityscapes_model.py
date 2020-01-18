import tensorflow as tf

from gscnn.model_definition import GSCNN
from cityscapes.train_and_evaluate import Trainer
import cityscapes.dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

batch_size = 4
network_input_h = network_input_w = 800
max_crop_downsample = 0.5
colour_aug_factor = 0.25
mixup_val = None
lr = 0.0001
l1 = 1.
l2 = 2.
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
optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

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





