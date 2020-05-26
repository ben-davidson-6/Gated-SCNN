import tensorflow as tf
import gated_shape_cnn.training.train_and_evaluate as gscnn_trainer
import gated_shape_cnn.model.model_definition
from unittest import mock
import unittest
#
# import numpy as np
#
# tf.random.set_seed(1)
# tf.config.set_visible_devices([], 'GPU')
#
#
# class TestTrainer(tf.test.TestCase):
#
#     def test_forward_pass(self,):
#         mock_model_prediction = np.random.random([1, 10, 10, 3])
#         mock_model_shape = np.random.random([1, 10, 10, 1])
#         model_out = np.concatenate([mock_model_prediction, mock_model_shape], axis=-1)
#         fake_model = mock.Mock(return_value=model_out)
#
#         mock_model_instance = mock.Mock()
#         mock_model_instance.model = fake_model
#         gscnn_trainer.Trainer.forward_pass(mock_model_instance, None, None, None)


if __name__ == '__main__':
    unittest.main()
