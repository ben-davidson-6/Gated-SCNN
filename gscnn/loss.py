import tensorflow as tf


def gen_dice(y_true, y_pred, eps=0.):
    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts ** 2)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)


def _edge_mag(tensor):
    tensor_edge = tf.image.sobel_edges(tensor)
    mag = tf.reduce_sum(tensor_edge**2, axis=-1, keepdims=True)
    mag /= tf.maximum(tf.reduce_max(mag, axis=-1, keepdims=True), 1.)
    return mag


def _gumbel_softmax(logits, eps=1e-8, tau=1.):
    g = tf.random.uniform(tf.shape(logits))
    g = -tf.math.log(eps - tf.math.log(g + eps))
    return tf.nn.softmax((logits + g)/tau)


def segmentation_edge_loss(gt_tensor, pred_tensor, thresh=0.8):

    pred_tensor = _gumbel_softmax(pred_tensor)
    gt_edges = _edge_mag(gt_tensor)
    pred_edges = _edge_mag(pred_tensor)

    gt_edges = tf.reshape(gt_edges, [-1, tf.shape(gt_edges)[-1]])
    pred_edges = tf.reshape(pred_edges, [-1, tf.shape(gt_edges)[-1]])

    edge_difference = tf.abs(gt_edges - pred_edges)

    mask_gt = tf.cast((gt_edges > thresh**2), tf.float32)
    contrib_0 = tf.reduce_mean(tf.boolean_mask(edge_difference, mask_gt))
    mask_pred = tf.stop_gradient(tf.cast((pred_edges > thresh**2), tf.float32))
    contrib_1 = tf.reduce_mean(tf.boolean_mask(edge_difference, mask_pred))

    return tf.reduce_mean(0.5*contrib_0 + 0.5*contrib_1)


def shape_edge_loss(gt_tensor, pred_tensor, pred_shape_tensor, thresh=0.8):
    mask = tf.cast(pred_shape_tensor > thresh, tf.float32)
    gt_tensor *= mask
    pred_tensor *= mask
    return gen_dice(gt_tensor, pred_tensor)


@tf.function
def loss(gt_tensor, pred_tensor, pred_shape_tensor):
    dice_loss = gen_dice(gt_tensor, pred_tensor)
    seg_edge = segmentation_edge_loss(gt_tensor, pred_tensor)
    edge_edge = shape_edge_loss(gt_tensor, pred_tensor, pred_shape_tensor)
    return dice_loss #+ seg_edge + edge_edge






