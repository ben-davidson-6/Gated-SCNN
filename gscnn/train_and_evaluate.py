import tensorflow as tf
import os
import gscnn.loss as gscnn_loss


class Trainer:
    def __init__(self, model,  train_dataset, val_dataset, epochs, optimiser, log_dir, model_dir):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.optimiser = optimiser
        self.step = tf.Variable(0, name='global_step', dtype=tf.int64)

        train_log_dir = os.path.join(log_dir, 'train')
        val_log_dir = os.path.join(log_dir, 'val')
        self.train_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_writer = tf.summary.create_file_writer(val_log_dir)
        self.epoch_train_loss = tf.keras.metrics.Mean('epoch_loss', dtype=tf.float32)
        self.epoch_val_loss = tf.keras.metrics.Mean('epoch_loss', dtype=tf.float32)

        self.log_freq = 100
        self.model_dir = model_dir

    def train_step(self, x, y):
        self.step.assign_add(1)
        with tf.GradientTape() as tape:
            prediction, pred_shape = self.model(x)
            loss = gscnn_loss.loss(y, prediction, pred_shape)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def log_batch_loss(self, loss):
        # update mean loss for epoch
        self.epoch_train_loss(loss)

        # update batch loss
        with self.train_writer.as_default():
            if tf.math.mod(self.step, self.log_freq) == 0 or tf.equal(self.step, 1):
                tf.summary.scalar('batch_loss', loss, step=self.step)
                tf.print('step ', self.step, ' batch loss ', loss, )

    def log_epoch_loss(self, epoch):
        with self.train_writer.as_default():
            tf.summary.scalar('Epoch loss', self.epoch_train_loss.result(), step=epoch)
        tf.print('----- Epoch loss ------ ', self.epoch_train_loss.result())

    @tf.function
    def train_epoch(self, epoch):
        for step, (x, y) in enumerate(self.train_dataset):
            loss = self.train_step(x, y)
            self.log_batch_loss(loss)

        self.log_epoch_loss(epoch)

    @tf.function
    def val_epoch(self, epoch):
        for step, (x, y) in enumerate(self.train_dataset):
            prediction, pred_shape = self.model(x)
            loss = gscnn_loss.loss(y, prediction, pred_shape)
            self.epoch_val_loss(loss)

        with self.val_writer.as_default():
            tf.summary.scalar('Epoch loss ', self.epoch_val_loss.result(), step=epoch)
        tf.print('----- val Epoch loss ------ ', self.epoch_val_loss.result())
        self.epoch_val_loss.reset_states()

    def make_weight_path(self, epoch):
        return os.path.join(self.model_dir, 'epoch_{}_val_loss_{.3f}'.format(epoch, self.epoch_val_loss.result()))

    def train_loop(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.val_epoch(epoch)
            self.model.save_weights(
                self.make_weight_path(epoch),
                save_format='tf')
            self.epoch_val_loss.reset_states()



