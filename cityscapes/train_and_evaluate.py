import tensorflow as tf
import os
import cityscapes.utils
import cityscapes
import gscnn.loss as gscnn_loss
from time import time


class Trainer:
    def __init__(self, model,  train_dataset, val_dataset, epochs, optimiser, log_dir, model_dir, l1, l2, l3, l4):
        self.weights = tf.constant([l1, l2, l3, l4])
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.optimiser = optimiser
        self.train_step_counter = tf.Variable(0, name='step_train', dtype=tf.int64)
        self.val_step_counter = tf.Variable(0, name='step_val', dtype=tf.int64)
        self.training = tf.Variable(True, name='training', dtype=tf.bool)

        train_log_dir = os.path.join(log_dir, 'train')
        val_log_dir = os.path.join(log_dir, 'val')
        self.train_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_writer = tf.summary.create_file_writer(val_log_dir)
        self.log_freq = 200
        self.model_dir = model_dir
        self.strategy = tf.distribute.get_strategy()

        # will build summaries in forward pass
        self.recorded_tensors = {
            'image': None,
            'label': None,
            'edge_label': None,
            'pred_label': None,
            'pred_shape': None,
            'seg_loss': None,
            'edge_loss': None,
            'edge_consistency': None,
            'edge_class_consistency': None,
            'loss': None,
            'accuracy': None,
        }
        self.epoch_metrics = {
            'accuracy': tf.keras.metrics.Mean(),
            'loss': tf.keras.metrics.Mean(),
            'mean_iou': tf.keras.metrics.MeanIoU(num_classes=cityscapes.N_CLASSES)}

        self.best_iou = -1.

    @tf.function
    def calculate_log_tensors(self, logits, label, seg_loss, edge_loss, edge_class_consistency, edge_consistency):
        keep_mask = tf.reduce_any(label == 1., axis=-1)

        flat_label = tf.argmax(label, axis=-1)
        flat_pred_label = tf.argmax(tf.nn.softmax(logits), axis=-1)

        loss = seg_loss + edge_loss + edge_class_consistency + edge_consistency

        flat_label_masked = flat_label[keep_mask]
        flat_pred_label_masked = flat_pred_label[keep_mask]
        correct = tf.reduce_sum(tf.cast(flat_label_masked == flat_pred_label_masked, tf.float32))
        total_vals = tf.shape(tf.reshape(flat_pred_label_masked, [-1]))[0]
        accuracy = correct / tf.cast(total_vals, tf.float32)
        return accuracy, loss, flat_label_masked, flat_pred_label_masked, flat_label, flat_pred_label

    @tf.function
    def calculate_images(self, flat_label, flat_pred_label):
        colour_array = tf.constant(cityscapes.TRAINING_COLOUR_PALETTE)
        label_image = tf.gather(colour_array, flat_label)
        pred_label_image = tf.gather(colour_array, flat_pred_label)
        return label_image, pred_label_image

    def log_pass(self, im, label, edge_label, logits, shape_head, seg_loss, edge_loss, edge_class_consistency, edge_consistency):
        step = self.train_step_counter if self.training else self.val_step_counter
        with tf.summary.record_if(tf.logical_and(self.training, tf.equal(tf.math.mod(step, self.log_freq), 0))):
            accuracy, loss, flat_label_masked, flat_pred_label_masked, flat_label, flat_pred_label = self.calculate_log_tensors(
                logits, label, seg_loss, edge_loss, edge_class_consistency, edge_consistency)

            self.epoch_metrics['accuracy'].update_state(accuracy)
            self.epoch_metrics['loss'].update_state(loss)
            self.epoch_metrics['mean_iou'].update_state(flat_label_masked, flat_pred_label_masked)

            # these do not work as intended https://github.com/tensorflow/tensorflow/issues/28007
            # with tf.summary.record_if(tf.equal(tf.math.mod(step, self.log_freq*5), 0)):
            #     label_image, pred_label_image = self.calculate_images(flat_label, flat_pred_label)
            #
            #     tf.summary.image(
            #         'edge_comparison',
            #         tf.concat([edge_label[..., 1:], shape_head], axis=2),
            #         step=step,
            #         max_outputs=1)
            #     tf.summary.image(
            #         'label_comparison',
            #         tf.concat([tf.cast(im, tf.uint8), label_image, pred_label_image], axis=2),
            #         step=step,
            #         max_outputs=1)
            tf.summary.scalar('loss/seg_loss', seg_loss, step=step)
            tf.summary.scalar('loss/edge_loss', edge_loss, step=step)
            tf.summary.scalar('loss/edge_class_consistency', edge_class_consistency, step=step)
            tf.summary.scalar('loss/edge_consistency', edge_consistency, step=step)
            tf.summary.scalar('loss/batch_loss', loss, step=step)
            tf.summary.scalar('batch_accuracy', accuracy, step=step)

    def forward_pass(self, im, label, edge_label):
        out = self.model(im, training=self.training)
        prediction, shape_head = out[..., :-1], out[..., -1:]
        seg_loss, edge_loss, edge_class_consistency, edge_consistency = gscnn_loss.loss(
            label, prediction, shape_head, edge_label, self.weights)
        self.log_pass(im, label, edge_label, prediction, shape_head, seg_loss, edge_loss, edge_class_consistency, edge_consistency)
        sub_losses = seg_loss, edge_loss, edge_class_consistency, edge_consistency
        return prediction, shape_head, sub_losses

    def train_step(self, im, label, edge_label):
        with tf.GradientTape() as tape:
            prediction, shape_head, sub_losses = self.forward_pass(im, label, edge_label)
            loss = sum(sub_losses)
            loss /= self.strategy.num_replicas_in_sync
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.model.trainable_variables))
        seg_loss, edge_loss, edge_class_consistency, edge_consistency = sub_losses
        self.log_pass(im, label, edge_label, prediction, shape_head, seg_loss, edge_loss, edge_class_consistency, edge_consistency)

    def log_metrics(self, epoch):
        for k in self.epoch_metrics:
            tf.summary.scalar('epoch_' + k, self.epoch_metrics[k].result(), step=epoch)
            self.epoch_metrics[k].reset_states()

    @tf.function
    def train_epoch(self, ):
        self.training.assign(True)
        with self.train_writer.as_default():
            for im, label, edge_label in self.train_dataset:
                self.strategy.experimental_run_v2(
                    self.train_step, args=(im, label, edge_label))
                self.train_step_counter.assign_add(1)

    @tf.function
    def val_epoch(self,):
        self.training.assign(False)
        with self.val_writer.as_default():
            for im, label, edge_label in self.val_dataset:
                self.strategy.experimental_run_v2(
                    self.forward_pass, args=(im, label, edge_label))
                self.val_step_counter.assign_add(1)

    def make_weight_path(self,):
        return os.path.join(self.model_dir, 'best')

    def train(self, epoch,):
        print('Training')
        self.train_epoch()
        with self.train_writer.as_default():
            self.log_metrics(epoch=epoch)

    def validate(self, epoch):
        self.val_epoch()
        self.model.save_weights(
            os.path.join(self.model_dir, 'latest'),
            save_format='tf')
        if self.epoch_metrics['mean_iou'].result() > self.best_iou:
            self.model.save_weights(
                self.make_weight_path(),
                save_format='tf')
            self.best_iou = self.epoch_metrics['mean_iou'].result()
        with self.val_writer.as_default():
            self.log_metrics(epoch=epoch)
        print('____ {} ____'.format(self.best_iou))

    def train_loop(self):
        for epoch in range(self.epochs):
            st = time()
            print('Epoch {}'.format(epoch))
            self.train(epoch,)
            print('Training an epoch took {0:1.0f}'.format(time() - st))

            print('Validating')
            st = time()
            self.validate(epoch)
            print('Validating an epoch took {0:1.0f}'.format(time() - st))
