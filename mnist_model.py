import tensorflow as tf
import numpy as np
from gated_shape_cnn import GSCNN
from loss import gen_dice
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Preprocess the data (these are Numpy arrays)
x_train = x_train.astype('float32') / 255
x_train = x_train[..., None]


def preprocess(xx, yy):
    # xx = tf.image.resize(xx, (75, 75))
    yy = tf.where(xx > 0.1, yy, 10)[..., 0]
    yy = tf.one_hot(yy, 11)
    return xx - 0.5, yy


x_val = x_train[-1000:]
y_val = y_train[-1000:]
x_train = x_train[:-1000]
y_train = y_train[:-1000]
train_d = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_d = train_d.map(preprocess)
train_d = train_d.repeat(5)
train_d = train_d.batch(64)

val_d = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_d = val_d.map(preprocess)
val_d = val_d.batch(64)


model = GSCNN(n_classes=11)
model.compile(loss=gen_dice)
model.fit(train_d)


for x, y in val_d:
    y_ = model.predict(x,)
    y_ = tf.nn.softmax(y_)
    plt.imshow(np.argmax(y_[0], axis=-1))
    plt.show()

    plt.imshow(np.argmax(y[0], axis=-1))
    plt.show()


