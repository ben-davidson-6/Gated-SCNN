import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from gscnn.model_definition import GSCNN
from cityscapes.train_and_evaluate import Trainer
import cityscapes.dataset


batch_size = 16
network_input_h = network_input_w = 180
max_crop_downsample = 0.75
colour_aug_factor = 0.25
lr = 0.001
l1 = 1.
l2 = 3.
l3 = 1.
l4 = 1.

cityscapes_dataset_loader = cityscapes.dataset.CityScapes(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    data_dir='/home/ben/datasets/cityscapes'
)

model = GSCNN(n_classes=cityscapes.N_CLASSES)

trainer = Trainer(
    model,
    cityscapes_dataset_loader.build_training_dataset(),
    cityscapes_dataset_loader.build_validation_dataset(),
    epochs=300,
    optimiser=tf.keras.optimizers.RMSprop(lr),
    log_dir='logs',
    model_dir='logs/model',
    l1=l1, l2=l2, l3=l3, l4=l4)
trainer.train_loop()





