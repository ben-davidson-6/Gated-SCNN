import tensorflow as tf

import numpy as np
from gscnn.model_definition import DebugModel
from scene_parsing_data.train_and_evaluate import Trainer
import matplotlib.pyplot as plt


def preprocess(xx, yy):
    yy = tf.where(xx > 0.1, yy, 10)[..., 0]
    yy = tf.one_hot(yy, 11)
    return xx - 0.5, yy


def build_datasets():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_train = x_train[..., None]

    x_val = x_train[-1000:]
    y_val = y_train[-1000:]
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]
    train_d = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_d = train_d.map(preprocess)
    train_d = train_d.batch(16)

    val_d = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_d = val_d.map(preprocess)
    val_d = val_d.batch(16)

    return train_d, val_d


def train_model():
    train_d, val_d = build_datasets()
    model = DebugModel()
    trainer = Trainer(
        model,
        train_d,
        val_d,
        epochs=2,
        optimiser=tf.keras.optimizers.RMSprop(),
        log_dir='logs',
        model_dir='logs/model')
    trainer.train_loop()


def display_model(fpath):
    train_d, val_d = build_datasets()
    model = DebugModel()
    model.load_weights(fpath)
    for x, y in val_d:
        pred, _ = model.predict(x)
        pred = tf.nn.softmax(pred[0])
        plt.imshow(np.argmax(pred, axis=-1))
        plt.show()


if __name__ == '__main__':
    train_model()
    display_model('./logs/model/epoch_1_val_loss_0.15999653935432434')





