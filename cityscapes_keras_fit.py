import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from gscnn.model_definition import GSCNN
from gscnn.loss import loss
import cityscapes.dataset


batch_size = 4
network_input_h = network_input_w = 800
max_crop_downsample = 0.9
colour_aug_factor = 0.25
mixup_val = None
lr = 0.001
l1 = 1.
l2 = 10.
l3 = 1.
l4 = 1.


class MeanIOU(tf.keras.metrics.Metric):

    def __init__(self, name='mean_iou', **kwargs):
        super(MeanIOU, self).__init__(name=name, **kwargs)
        self.mean_iou = tf.keras.metrics.MeanIoU(cityscapes.N_CLASSES)

    def update_state(self, y_true, y_pred, sample_weight=None):
        label, edge_label = y_true[..., :-2], y_true[..., -2:]
        logits, shape_head = y_pred[..., :-1], y_pred[..., -1:]

        keep_mask = tf.reduce_any(label == 1., axis=-1)
        label_flat = tf.argmax(label, axis=-1)
        pred_label_flat = tf.argmax(tf.nn.softmax(logits), axis=-1)
        label_flat = label_flat[keep_mask]
        pred_label_flat = pred_label_flat[keep_mask]

        self.mean_iou.update_state(label_flat, pred_label_flat)

    def result(self):
        return self.mean_iou.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mean_iou.reset_states()


def keras_loss(y_true, y_pred):
    pred_logit, pred_shape = y_pred[..., :-1], y_pred[..., -1:]
    label, edge_label = y_true[..., :-2], y_true[..., -2:]
    return loss(label, pred_logit, pred_shape, edge_label, [l1, l2, l3, l4])


cityscapes_dataset_loader = cityscapes.dataset.CityScapes(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    data_dir='/home/ben/datasets/cityscapes',
    mixup_val=mixup_val,
    merge_labels=True
)


n_train_images = 3750
n_steps_in_epoch = n_train_images // batch_size
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    0.001,
    n_steps_in_epoch*200,
    0.000001)
optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.5)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = GSCNN(n_classes=cityscapes.N_CLASSES)
    model.compile(
        optimizer=optimiser,
        loss=keras_loss,
        metrics=[MeanIOU()]
    )
model.fit(
    cityscapes_dataset_loader.build_training_dataset(),
    validation_data=cityscapes_dataset_loader.build_validation_dataset(),
    epochs=1000,
    callbacks=[
        tf.keras.callbacks.TensorBoard(write_graph=True),
        tf.keras.callbacks.ModelCheckpoint('./logs/model', monitor='val_mean_iou', save_best_only=True, mode='max')]
    )





