import tensorflow as tf
from gated_shape_cnn.model.layers import gradient_mag


def _generalised_dice(y_true, y_pred, eps=0.0, from_logits=True):
    """
    :param y_true [b, h, w, c]:
    :param y_pred [b, h, w, c]:
    :param eps weight fudge factor for zero counts:
    :return generalised dice loss:

    see https://www.nature.com/articles/s41598-018-26350-3
    """

    # [b, h, w, classes]
    if from_logits:
        y_pred = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(y_pred, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / counts**2
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights * multed, axis=-1)
    denom = tf.reduce_sum(weights * summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)


def _gumbel_softmax(logits, eps=1e-8, tau=1.):
    """

    :param logits:
    :param eps:
    :param tau temprature:
    :return soft approximation to argmax:

    see https://arxiv.org/abs/1611.01144
    """
    g = tf.random.uniform(tf.shape(logits))
    g = -tf.math.log(eps - tf.math.log(g + eps))
    return tf.nn.softmax((logits + g) / tau)


def _segmentation_edge_loss(gt_tensor, logit_tensor, thresh=0.8):
    """

    :param gt_tensor [b, h, w, c] segmentation labels:
    :param pred_tensor [b, h, w, c] segmentation logits:
    :param thresh intensity to be considered edge:
    :return the difference in boundaries between predicted versus actual
            where the boundaries come from the segmentation, rather than
            the shape head:
    """

    # soft approximation to argmax, so we can build an edge
    logit_tensor = _gumbel_softmax(logit_tensor)

    # normalised image gradients to give us edges
    # images will be [b, h, w, n_classes]
    gt_edges = gradient_mag(gt_tensor)
    pred_edges = gradient_mag(logit_tensor)

    # [b*h*w, n]
    gt_edges = tf.reshape(gt_edges, [-1, tf.shape(gt_edges)[-1]])
    pred_edges = tf.reshape(pred_edges, [-1, tf.shape(gt_edges)[-1]])

    # take the difference between these two gradient magnitudes
    # we will first take all the edges from the ground truth image
    # and then all the edges from the predicted
    edge_difference = tf.abs(gt_edges - pred_edges)

    # gt edges and disagreement with pred
    mask_gt = tf.cast((gt_edges > thresh ** 2), tf.float32)
    contrib_0 = tf.boolean_mask(edge_difference, mask_gt)

    contrib_0 = tf.cond(
        tf.greater(tf.size(contrib_0), 0),
        lambda: tf.reduce_mean(contrib_0),
        lambda: 0.)

    # vice versa
    mask_pred = tf.stop_gradient(tf.cast((pred_edges > thresh ** 2), tf.float32))
    contrib_1 = tf.reduce_mean(tf.boolean_mask(edge_difference, mask_pred))
    contrib_1 = tf.cond(
        tf.greater(tf.size(contrib_1), 0),
        lambda: tf.reduce_mean(contrib_1),
        lambda: 0.)
    return tf.reduce_mean(0.5 * contrib_0 + 0.5 * contrib_1)


def _shape_edge_loss(gt_tensor, pred_tensor, pred_shape_tensor, keep_mask, thresh=0.8):
    """
    :param gt_tensor [b, h, w, c]:
    :param pred_tensor [b, h, w, c]:
    :param pred_shape_tensor [b, h, w, 1]:
    :param keep_mask binary mask of pixels to keep (eg in cityscapes we ignore 255):
    :param thresh probability to consider an edge in our prediction:
    :return cross entropy of classifications near on an edge:

     whereever we have predicted an edge, calculated the cross entropy there.
    This penalises the edges more strongly, encouraging them to be correct at the boundary
    """

    # where we have predicted an edge and which are pixels
    # we care about
    mask = pred_shape_tensor > thresh
    mask = tf.stop_gradient(mask[..., 0])
    mask = tf.logical_and(mask, keep_mask)

    # get relavent predicitons and truth
    gt = gt_tensor[mask]
    pred = pred_tensor[mask]

    # cross entropy, we may not have any edges, in which case return 0
    if tf.reduce_sum(tf.cast(mask, tf.float32)) > 0:
        return tf.reduce_mean(tf.losses.categorical_crossentropy(gt, pred, from_logits=True))
    else:
        return 0.


def _weighted_cross_entropy(y_true, y_pred, keep_mask):
    """

    :param y_true [b, h, w, c]:
    :param y_pred [b, h, w, c]:
    :return weighted cross entropy:
    """

    # ignore zertain pixels
    # makes both tensors [n, c]
    y_true = y_true[keep_mask]
    y_pred = y_pred[keep_mask]

    # weights
    rs = tf.reduce_sum(y_true, axis=0, keepdims=True)
    N = tf.cast(tf.shape(y_true)[0], tf.float32)
    weights = (N - rs)/N + 1

    # everything here is one hot so this essentially picks the class weight
    # per row of y_true
    weights = tf.reduce_sum(y_true*weights, axis=1)

    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    weighted_losses = unweighted_losses * weights
    loss = tf.reduce_mean(weighted_losses)
    return loss


def loss(gt_label, logits, shape_head, edge_label, loss_weights):
    tf.debugging.assert_shapes([
        (gt_label,     ('b', 'h', 'w', 'c')),
        (logits,       ('b', 'h', 'w', 'c')),
        (shape_head,   ('b', 'h', 'w', 1)),
        (edge_label,   ('b', 'h', 'w', 2)),
        (loss_weights, (4,))],)

    # in cityscapes we ignore some classes, which means that there will
    # be pixels without any class
    keep_mask = tf.reduce_any(gt_label == 1., axis=-1)
    anything_active = tf.reduce_any(keep_mask)

    # standard weighted cross entropy
    # we weight each class by 1 + (1 - batch_prob_of_class)
    # where we get the prob by counting ground truth pixels
    seg_loss = tf.cond(
        anything_active,
        lambda: _weighted_cross_entropy(gt_label, logits, keep_mask) * loss_weights[0],
        lambda: 0.)

    # Generalised dice loss on the edges predicted by the network
    shape_probs = tf.concat([1. - shape_head, shape_head], axis=-1)
    edge_loss = _generalised_dice(edge_label, shape_probs) * loss_weights[1]

    # regularizing loss
    # this ensures that the edges themselves are consistent
    edge_consistency = _segmentation_edge_loss(gt_label, logits) * loss_weights[2]
    # this ensures that the classifcatiomn at the edges is correct
    edge_class_consistency = tf.cond(
        anything_active,
        lambda: _shape_edge_loss(gt_label, logits, shape_head, keep_mask) * loss_weights[3],
        lambda: 0.)
    return seg_loss, edge_loss, edge_class_consistency, edge_consistency

