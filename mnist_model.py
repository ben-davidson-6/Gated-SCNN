import tensorflow as tf
import numpy as np
from gscnn.model_definition import DebugModel
from gscnn.train_and_evaluate import Trainer
import matplotlib.pyplot as plt


def build_datasets():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_train = x_train[..., None]


    def preprocess(xx, yy):
        yy = tf.where(xx > 0.1, yy, 10)[..., 0]
        yy = tf.one_hot(yy, 11)
        return xx - 0.5, yy


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

if __name__ == '__main__':
    train_d, val_d = build_datasets()
    model = DebugModel()
    trainer = Trainer(
        model,
        train_d,
        val_d,
        epochs=1,
        optimiser=tf.keras.optimizers.RMSprop(),
        log_dir='logs',
        model_dir='logs/model')
    trainer.train_loop()
    for x, y in val_d:
        pred, _ = model.predict(x)
        pred = tf.nn.softmax(pred[0])
        plt.imshow(np.argmax(pred, axis=-1))
        plt.show()




