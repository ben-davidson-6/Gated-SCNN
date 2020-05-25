import tensorflow as tf
import os
import sys

from time import time

import gated_shape_cnn
import gated_shape_cnn.training.loss as gscnn_loss

from gated_shape_cnn.training import utils
from gated_shape_cnn.model import GSCNN


class Trainer:
    LOG_FREQ = 200
    """
    Custom training loop in tensorflow 2. The loop is as follows:
    for n epochs
        train_epoch
            for batch in epoch
                accum_updates
                accum_updates += forward_backward_pass
                if number accumulations
                    weights += accum_updates
                    accum_updates = 0
        val epoch
            evaluate model
            if mean iou > best iou
                overwrite best model
            overwrite latest model
    """
    def __init__(
            self,
            model,
            train_dataset,
            val_dataset,
            epochs,
            optimiser,
            log_dir,
            model_dir,
            loss_weights,
            accumulation_iterations=None):
        """

        :param model GSCNN model:
        :param  train_dataset tf.data.Dataset, which when iterated returns
                image, label, edge, also should not repeat indefintely:
        :param val_dataset tf.data.Dataset, which when iterated returns
                image, label, edge, also should not repeat indefintely:
        :param epochs number epochs to go through the data:
        :param optimiser tf.keras.optimizer:
        :param log_dir where you want the logs to appear:
        :param model_dir where you want ot save the model:
        :param loss_weights tensor shape [4], gscnn loss params (lambda_i's from paper) :
        :param accumulation_iterations accumulate
               gradients over this many training steps, before updating weights:
        """
        self.loss_weights = tf.constant(loss_weights)
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.optimiser = optimiser

        # counters for tensorboard
        self.train_step_counter = tf.Variable(0, name='step_train', dtype=tf.int64, trainable=False)
        self.val_step_counter = tf.Variable(0, name='step_val', dtype=tf.int64, trainable=False)
        self.epoch = tf.Variable(0, name='epoch', dtype=tf.int64, trainable=False)
        self.start_of_epoch = tf.Variable(True, name='training', dtype=tf.bool, trainable=False)

        # flag used for inference versus training in model(x, self.training)
        # as well as some other things
        self.training = tf.Variable(True, name='training', dtype=tf.bool, trainable=False)
        self.n_accum_iters = accumulation_iterations

        if accumulation_iterations is not None:
            # TODO fix the hackyness here actually calling the model will need to define
            #  self.build for all the layers

            # build the model so that all unbuilt weights are added to the trainable
            # variables
            self.model(tf.zeros([1, 720, 720, 3],))

            # create an accumulation variable for every traininable variable in the model
            # we can use these to accumulate weight updates across training steps
            self.accum_vars = [
                tf.Variable(tf.zeros_like(tv.read_value()), trainable=False)
                for tv in self.model.trainable_variables]

            # counter for how many iterations have happened, when this reaches self.n_accum_iters
            # we update the weights and zero the accum vars
            self.current_iters = tf.Variable(0, trainable=False, dtype=tf.int32)
        else:
            self.accum_vars = None
            self.current_iters = None

        # where we are going to save the tensorboard stuff
        train_log_dir = os.path.join(log_dir, 'train')
        val_log_dir = os.path.join(log_dir, 'val')
        self.train_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_writer = tf.summary.create_file_writer(val_log_dir)
        self.model_dir = model_dir

        # track these metrics across the whole epoch
        self.epoch_metrics = {
            'accuracy': tf.keras.metrics.Accuracy(),
            'loss': tf.keras.metrics.Mean(),
            'mean_iou': tf.keras.metrics.MeanIoU(num_classes=self.model.n_classes)}

        # initialise the best iou so far
        # will save best model according to this
        self.best_iou = -1.

    #######################################################################
    # Forward and backward passes
    #######################################################################

    def forward_pass(self, im, label, edge_label):
        # forward through model
        out = self.model(im, training=self.training)
        prediction, shape_head = out[..., :-1], out[..., -1:]

        # calculate the loss, consists of several components
        sub_losses = gscnn_loss.loss(
            label,
            prediction,
            shape_head,
            edge_label,
            self.loss_weights)
        return prediction, shape_head, sub_losses

    def log_pass(self, im, label, edge_label, prediction, shape_head, sub_losses):
        # log to tensorboard
        flat_label, flat_pred, keep_mask = self.log_images(im, label, edge_label, prediction, shape_head)
        loss = self.log_loss(sub_losses)
        self.update_metrics(flat_label, flat_pred, loss, keep_mask)
        return loss

    def apply_gradients(self, gradients):
        self.train_step_counter.assign_add(1)
        if self.n_accum_iters is None:
            self.optimiser.apply_gradients(
                zip(gradients, self.model.trainable_variables))
        else:
            # accumulate gradients
            self.current_iters.assign_add(1)
            for k, grad in enumerate(gradients):
                self.accum_vars[k].assign_add(grad / self.n_accum_iters)

            # if accumulated_enough
            if self.current_iters == self.n_accum_iters:
                # apply gradients
                self.optimiser.apply_gradients(zip(self.accum_vars, self.model.trainable_variables))

                # reset
                self.current_iters.assign(0)
                for k, _ in enumerate(self.accum_vars):
                    self.accum_vars[k].assign(tf.zeros_like(self.accum_vars[k]))

    def train_step(self, im, label, edge_label):
        with tf.GradientTape() as tape:
            prediction, shape_head, sub_losses = self.forward_pass(im, label, edge_label)
            loss = self.log_pass(im, label, edge_label, prediction, shape_head, sub_losses)
            regularization_loss = tf.add_n(self.model.losses)
            loss += regularization_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.apply_gradients(gradients)

    #######################################################################
    # Training loop
    #######################################################################

    def train_loop(self):
        for _ in range(self.epochs):
            print('Epoch {}'.format(self.epoch.numpy()))
            print('Training')
            st = time()
            self.train()
            print('\t took {0:1.0f} seconds'.format(time() - st))

            print('Validation \r')
            v_st = time()
            self.validate()
            self.epoch.assign_add(1)
            print('\t took {0:1.0f} seconds'.format(time() - v_st))

            print('Total time for epoch took {0:1.0f}'.format(time() - st))
            print('**********************************')

    def train(self,):
        self.training.assign(True)
        with self.train_writer.as_default():
            self.train_epoch()
            self.log_metrics()

    def save_model(self):
        self.model.save_weights(os.path.join(self.model_dir, 'latest'), save_format='tf')
        if self.epoch_metrics['mean_iou'].result() > self.best_iou:
            self.model.save_weights(self.make_weight_path(), save_format='tf')
            self.best_iou = self.epoch_metrics['mean_iou'].result()

    def validate(self):
        self.training.assign(False)
        with self.val_writer.as_default():
            self.val_epoch()
            self.save_model()
            self.log_metrics()

    @staticmethod
    def validate_data(im, label, edge):
        utils.validate_image_tensor(im)
        utils.validate_label_tensor(label)
        utils.validate_edge_tensor(edge)

    @tf.function
    def train_epoch(self, ):
        self.start_of_epoch.assign(True)
        for im, label, edge_label in self.train_dataset:
            Trainer.validate_data(im, label, edge_label)
            self.train_step(im, label, edge_label)

    @tf.function
    def val_epoch(self, ):
        self.start_of_epoch.assign(True)
        for im, label, edge_label in self.val_dataset:
            Trainer.validate_data(im, label, edge_label)
            prediction, shape_head, sub_losses = self.forward_pass(im, label, edge_label)
            self.log_pass(im, label, edge_label, prediction, shape_head, sub_losses)
            self.val_step_counter.assign_add(1)

    #######################################################################
    # Logging
    #######################################################################

    def get_step(self):
        return self.train_step_counter if self.training else self.val_step_counter

    def log_images(self, image, label, edge_label, prediction, shape_head):
        """save some images at the start of every epoch to tensorboard"""
        colour_array = tf.constant(gated_shape_cnn.COLOUR_PALLETTE)
        keep_mask = tf.reduce_any(label == 1., axis=-1)
        flat_label = tf.argmax(label, axis=-1)
        flat_label = tf.where(keep_mask, flat_label, tf.cast(gated_shape_cnn.N_COLOURS - 1, tf.int64), )
        flat_pred = tf.argmax(prediction, axis=-1)
        with tf.summary.record_if(self.start_of_epoch.value()):
            self.start_of_epoch.assign(False)
            # edges
            tf.summary.image(
                'edge_comparison',
                tf.concat([edge_label[..., 1:], shape_head], axis=2),
                step=self.epoch,
                max_outputs=1)

            # segmentation
            label_image = tf.gather(colour_array, flat_label)
            pred_label_image = tf.gather(colour_array, flat_pred)
            tf.summary.image(
                'label_comparison',
                tf.concat([tf.cast(image, tf.uint8), label_image, pred_label_image], axis=2),
                step=self.epoch,
                max_outputs=1)
        return flat_label, flat_pred, keep_mask

    def log_loss(self, sub_losses):
        """save the various losses to tensorboard and sum them"""
        step = self.get_step()
        log_condition = tf.equal(
            tf.math.mod(step, Trainer.LOG_FREQ),
            0)
        loss = tf.add_n(sub_losses)
        with tf.summary.record_if(log_condition):
            seg_loss, edge_loss, edge_class_consistency, edge_consistency = sub_losses
            tf.summary.scalar('loss/seg_loss', seg_loss, step=step)
            tf.summary.scalar('loss/edge_loss', edge_loss, step=step)
            tf.summary.scalar('loss/edge_class_consistency', edge_class_consistency, step=step)
            tf.summary.scalar('loss/edge_consistency', edge_consistency, step=step)
            tf.summary.scalar('loss/total_loss', loss, step=step)
        return loss

    def log_metrics(self,):
        for k in self.epoch_metrics:
            tf.summary.scalar('epoch_' + k, self.epoch_metrics[k].result(), step=self.epoch)
            tf.print(
                '\t epoch metric {}: '.format(k), self.epoch_metrics[k].result(),
                output_stream=sys.stdout,)
            self.epoch_metrics[k].reset_states()

    def update_metrics(self, flat_label, flat_pred, loss, keep_mask):
        """calulate epoch level information"""
        flat_label_masked = flat_label[keep_mask]
        flat_pred_masked = flat_pred[keep_mask]
        self.epoch_metrics['accuracy'].update_state(flat_label_masked, flat_pred_masked)
        self.epoch_metrics['loss'].update_state(loss)
        self.epoch_metrics['mean_iou'].update_state(flat_label_masked, flat_pred_masked)

    def make_weight_path(self,):
        return os.path.join(self.model_dir, 'best')


def train_model(
        n_classes,
        train_data,
        val_data,
        optimiser,
        epochs,
        log_dir,
        model_dir,
        accum_iterations=4,
        loss_weights=(1., 20., 1., 1.)):

    # build the model
    model = GSCNN(n_classes=n_classes)

    # train
    trainer = Trainer(
        model,
        train_data,
        val_data,
        epochs=epochs,
        optimiser=optimiser,
        log_dir=log_dir,
        model_dir=model_dir,
        loss_weights=loss_weights,
        accumulation_iterations=accum_iterations,)
    trainer.train_loop()











