import tensorflow as tf


class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 center: bool = True,
                 scale: bool = True,
                 epsilon: float = 1e-3,
                 name: str = None,
                 normaxis: int = -1,
                 momentum=0.999,
                 **kwargs):
        """
        args:
        - center: bool
        use mean statistics
        - scale: bool
        use stddev statistics
        - epsilon: float
        epsilon for zero division
        - name: str
        layer's name
        - normaxis: int
        layer's feature axis
        if data is NHWC => C (-1)
        """
        super(BatchNormalization, self).__init__(
            name=name,
            **kwargs
        )
        self.axis = normaxis
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape: list):
        """
        args:
        - input_shape: list
        example. [None, H, W, C] = [None, 32, 32, 3] (cifer 10)
        """
        self.feature_dim = input_shape[self.axis]
        self.axes = list(range(len(input_shape)))
        self.axes.pop(self.axis)
        if self.scale:
            self.gamma = self.add_weight(
                shape=(self.feature_dim,),
                name='gamma',
                initializer='ones',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=(self.feature_dim,),
                name='beta',
                initializer='zeros',
            )
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.feature_dim,),
            initializer=tf.initializers.zeros,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)
        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=(self.feature_dim,),
            initializer=tf.initializers.ones,
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
            experimental_autocast=False)
        super(BatchNormalization, self).build(input_shape)

    def _assign_moving_average(self, variable: tf.Tensor, value: tf.Tensor):
        return variable.assign(variable * (1.0 - self.momentum)
                               + value * self.momentum)

    def call(self, x: tf.Tensor, training=True, **kwargs):
        if training:
            ctx = tf.distribute.get_replica_context()
            n = ctx.num_replicas_in_sync
            mean, mean_sq = ctx.all_reduce(
                tf.distribute.ReduceOp.SUM,
                [tf.reduce_mean(x, axis=self.axes) / n,
                 tf.reduce_mean(tf.square(x),
                                axis=self.axes) / n]
            )
            variance = mean_sq - mean ** 2
            mean_update = self._assign_moving_average(self.moving_mean, mean)
            variance_update = self._assign_moving_average(
                self.moving_variance, variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        z = tf.nn.batch_normalization(x, mean=mean,
                                      variance=variance,
                                      offset=self.beta,
                                      scale=self.gamma,
                                      variance_epsilon=self.epsilon)
        return z
