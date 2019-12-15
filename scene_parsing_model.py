import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


from gscnn.model_definition import GSCNN
from scene_parsing_data.train_and_evaluate import Trainer
import scene_parsing_data.dataset


batch_size = 8
network_input_h = network_input_w = 256
max_crop_downsample = 0.5
colour_aug_factor = 0.25

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
    epochs=1,
    optimiser=tf.keras.optimizers.RMSprop(0.0001),
    log_dir='logs',
    model_dir='logs/model')
trainer.train_loop()





