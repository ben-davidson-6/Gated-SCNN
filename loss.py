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
    mag = tf.linalg.norm(tensor_edge, axis=-1)
    mag /= tf.reduce_max(max, axis=-1, keepdims=True)
    return mag


def _gumbel_softmax(logits, eps=1e-8, tau=1.):
    g = tf.random.uniform(tf.shape(logits))
    g = -tf.math.log(eps - tf.math.log(g + eps))
    return tf.math.softmax((logits + g)/tau)


def segmentation_edge_loss(gt_tensor, pred_tensor, thresh=0.8):

    pred_tensor = _gumbel_softmax(pred_tensor)
    gt_edges = _edge_mag(gt_tensor)
    pred_edges = _edge_mag(pred_tensor)

    edge_difference = tf.math.abs(gt_edges - pred_edges)
    contrib_0 = (gt_edges > thresh)*edge_difference
    contrib_1 = (pred_edges > thresh)*edge_difference

    return 0.5*contrib_0 + 0.5*contrib_1


def shape_edge_loss(gt_tensor, pred_tensor, pred_shape_tensor, thresh=0.8):
    gt_tensor = gt_tensor[pred_shape_tensor > thresh]
    pred_tensor = pred_tensor[pred_shape_tensor > thresh]
    return generalised_dice(gt_tensor, pred_tensor)


def loss(gt_tensor, pred_tensor, pred_shape_tensor):
    dice_loss = generalised_dice(gt_tensor, pred_tensor)
    seg_edge = segmentation_edge_loss(gt_tensor, pred_tensor)
    edge_edge = shape_edge_loss(gt_tensor, pred_tensor, pred_shape_tensor)
    return dice_loss + seg_edge + edge_edge


def loss_wrapper(pred_shape_tensor):
    def f(y_true, y_pred):
        return loss(y_true, y_pred, pred_shape_tensor)
    return f




