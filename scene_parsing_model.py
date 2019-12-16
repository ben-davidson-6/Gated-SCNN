import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from gscnn.model_definition import GSCNN
from scene_parsing_data.train_and_evaluate import Trainer
import scene_parsing_data.dataset


batch_size = 8
network_input_h = network_input_w = 256
max_crop_downsample = 0.75
colour_aug_factor = 0.25
lr = 0.001
l1 = 1.
l2 = 1.
l3 = 1.
l4 = 1.

scene_data = scene_parsing_data.dataset.SceneParsingDataset(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor
)

model = GSCNN(n_classes=scene_parsing_data.N_CLASSES)

trainer = Trainer(
    model,
    scene_data.build_training_dataset(),
    scene_data.build_validation_dataset(),
    epochs=300,
    optimiser=tf.keras.optimizers.RMSprop(lr),
    log_dir='logs',
    model_dir='logs/model',
    l1=l1, l2=l2, l3=l3, l4=l4)
trainer.train_loop()





