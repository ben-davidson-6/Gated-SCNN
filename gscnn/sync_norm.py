from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.distribute import reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils


def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
    return nn.moments(inputs, reduction_axes, keep_dims=keep_dims)


def _moments(self, inputs, reduction_axes, keep_dims):
    mean, variance = self._calculate_mean_and_var(inputs, reduction_axes,
                                                  keep_dims)
    # TODO(b/129279393): Support zero batch input in non DistributionStrategy
    # code as well.
    if self._support_zero_size_input():
      inputs_size = array_ops.size(inputs)
      mean = array_ops.where(inputs_size > 0, mean, K.zeros_like(mean))
      variance = array_ops.where(inputs_size > 0, variance,
                                 K.zeros_like(variance))
    return mean, variance


def call(self, inputs, training=None):
    training = self._get_training_value(training)

    if self.virtual_batch_size is not None:
      # Virtual batches (aka ghost batches) can be simulated by reshaping the
      # Tensor and reusing the existing batch norm implementation
      original_shape = [-1] + inputs.shape.as_list()[1:]
      expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

      # Will cause errors if virtual_batch_size does not divide the batch size
      inputs = array_ops.reshape(inputs, expanded_shape)

      def undo_virtual_batching(outputs):
        outputs = array_ops.reshape(outputs, original_shape)
        return outputs

    if self.fused:
      outputs = self._fused_batch_norm(inputs, training=training)
      if self.virtual_batch_size is not None:
        # Currently never reaches here since fused_batch_norm does not support
        # virtual batching
        outputs = undo_virtual_batching(outputs)
      return outputs

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.constant_value(training)
    if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
      mean, variance = self.moving_mean, self.moving_variance
    else:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = tf_utils.smart_cond(training,
                                        lambda: adj_scale,
                                        lambda: array_ops.ones_like(adj_scale))
        adj_bias = tf_utils.smart_cond(training,
                                       lambda: adj_bias,
                                       lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      mean, variance = self._moments(
          math_ops.cast(inputs, self._param_dtype),
          reduction_axes,
          keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training,
                                 lambda: mean,
                                 lambda: ops.convert_to_tensor(moving_mean))
      variance = tf_utils.smart_cond(
          training,
          lambda: variance,
          lambda: ops.convert_to_tensor(moving_variance))

      if self.virtual_batch_size is not None:
        # This isn't strictly correct since in ghost batch norm, you are
        # supposed to sequentially update the moving_mean and moving_variance
        # with each sub-batch. However, since the moving statistics are only
        # used during evaluation, it is more efficient to just update in one
        # step and should not make a significant difference in the result.
        new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
        new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
      else:
        new_mean, new_variance = mean, variance

      if self._support_zero_size_input():
        inputs_size = array_ops.size(inputs)
      else:
        inputs_size = None
      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training, inputs_size)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        """Compute the updates for mean and variance."""
        return self._assign_moving_average(var, value, self.momentum,
                                           inputs_size)

      def mean_update():
        true_branch = lambda: _do_update(self.moving_mean, new_mean)
        false_branch = lambda: self.moving_mean
        return tf_utils.smart_cond(training, true_branch, false_branch)

      def variance_update():
        """Update the moving variance."""

        def true_branch_renorm():
          # We apply epsilon as part of the moving_stddev to mirror the training
          # code path.
          moving_stddev = _do_update(self.moving_stddev,
                                     math_ops.sqrt(new_variance + self.epsilon))
          return self._assign_new_value(
              self.moving_variance,
              # Apply relu in case floating point rounding causes it to go
              # negative.
              K.relu(moving_stddev * moving_stddev - self.epsilon))

        if self.renorm:
          true_branch = true_branch_renorm
        else:
          true_branch = lambda: _do_update(self.moving_variance, new_variance)

        false_branch = lambda: self.moving_variance
        return tf_utils.smart_cond(training, true_branch, false_branch)

      self.add_update(mean_update)
      self.add_update(variance_update)

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    if scale is not None:
      scale = math_ops.cast(scale, inputs.dtype)
    # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
    # math in float16 hurts validation accuracy of popular models like resnet.
    outputs = nn.batch_normalization(inputs,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     offset,
                                     scale,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    return outputs


import tensorflow.keras.layers

MonkeyPatchedBatchNormalisation = tensorflow.keras.layers.BatchNormalization
MonkeyPatchedBatchNormalisation._moments = _moments
MonkeyPatchedBatchNormalisation._calculate_mean_and_var = _calculate_mean_and_var
MonkeyPatchedBatchNormalisation.call = call


class SyncBatchNormalization(MonkeyPatchedBatchNormalisation):
  r"""Normalize and scale inputs or activations synchronously across replicas.
  Applies batch normalization to activations of the previous layer at each batch
  by synchronizing the global batch statistics across all devices that are
  training the model. For specific details about batch normalization please
  refer to the `tf.keras.layers.BatchNormalization` layer docs.
  If this layer is used when using tf.distribute strategy to train models
  across devices/workers, there will be an allreduce call to aggregate batch
  statistics across all replicas at every training step. Without tf.distribute
  strategy, this layer behaves as a regular `tf.keras.layers.BatchNormalization`
  layer.
  Example usage:
  ```
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.experimental.SyncBatchNormalization())
  ```
  Arguments:
    axis: Integer, the axis that should be normalized
      (typically the features axis).
      For instance, after a `Conv2D` layer with
      `data_format="channels_first"`,
      set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
      If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
      If False, `gamma` is not used.
      When the next layer is linear (also e.g. `nn.relu`),
      this can be disabled since the scaling
      will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    trainable: Boolean, if `True` the variables will be marked as trainable.
  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode.
      - `training=True`: The layer will normalize its inputs using the
        mean and variance of the current batch of inputs.
      - `training=False`: The layer will normalize its inputs using the
        mean and variance of its moving statistics, learned during training.
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    Same shape as input.
  """

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               trainable=True,
               adjustment=None,
               name=None,
               **kwargs):

    # Currently we only support aggregating over the global batch size.
    super(SyncBatchNormalization, self).__init__(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=False,
        trainable=trainable,
        virtual_batch_size=None,
        name=name,
        **kwargs)

  def _calculate_mean_and_var(self, x, axes, keep_dims):

    with ops.name_scope('moments', values=[x, axes]):
      # The dynamic range of fp16 is too limited to support the collection of
      # sufficient statistics. As a workaround we simply perform the operations
      # on 32-bit floats before converting the mean and variance back to fp16
      y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
      replica_ctx = ds.get_replica_context()
      if replica_ctx:
        local_sum = math_ops.reduce_sum(y, axis=axes, keepdims=True)
        local_squared_sum = math_ops.reduce_sum(math_ops.square(y), axis=axes,
                                                keepdims=True)
        batch_size = math_ops.cast(array_ops.shape_v2(y)[0], dtypes.float32)
        y_sum, y_squared_sum, global_batch_size = (
            replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, [
                local_sum, local_squared_sum, batch_size]))

        axes_vals = [(array_ops.shape_v2(y))[i] for i in range(1, len(axes))]
        multiplier = math_ops.cast(math_ops.reduce_prod(axes_vals),
                                   dtypes.float32)
        multiplier = multiplier * global_batch_size

        mean = y_sum / multiplier
        y_squared_mean = y_squared_sum / multiplier
        # var = E(x^2) - E(x)^2
        variance = y_squared_mean - math_ops.square(mean)
      else:
        # Compute true mean while keeping the dims for proper broadcasting.
        mean = math_ops.reduce_mean(y, axes, keepdims=True, name='mean')
        # sample variance, not unbiased variance
        # Note: stop_gradient does not change the gradient that gets
        #       backpropagated to the mean from the variance calculation,
        #       because that gradient is zero
        variance = math_ops.reduce_mean(
            math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
            axes,
            keepdims=True,
            name='variance')
      if not keep_dims:
        mean = array_ops.squeeze(mean, axes)
        variance = array_ops.squeeze(variance, axes)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(variance, dtypes.float16))
      else:
        return (mean, variance)