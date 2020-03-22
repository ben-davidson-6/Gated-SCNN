from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.distribute import reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

import tensorflow.keras.layers


class SyncBatchNormalization(tensorflow.keras.layers.BatchNormalization):

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


if __name__ == '__main__':
    import os
    import tensorflow as tf
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2"
    strategy = tf.distribute.MirroredStrategy()
    data = np.array([1., -1, 1, -1], dtype=np.float32)

    with strategy.scope():
        s = tf.keras.layers.BatchNormalization()
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(2)
        train_dataset = strategy.experimental_distribute_dataset(dataset)
        for b in train_dataset:
            strategy.experimental_run_v2(
                s, args=(b, True))
