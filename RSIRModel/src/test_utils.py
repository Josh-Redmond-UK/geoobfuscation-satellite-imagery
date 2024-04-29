import unittest
import tensorflow as tf
from utils import *
import numpy as np


class TestUtils(unittest.TestCase):

    def test_prepare_for_training(self):
        # Create a dummy dataset
        batch_size = 64
        dataset_size= 128
        dataset = tf.data.Dataset.from_tensor_slices({
            "image": tf.random.normal(shape=(dataset_size, 64, 64, 3)),
            "label": tf.random.uniform(shape=(dataset_size,), maxval=10, dtype=tf.int32)
        })

        # Test prepare_for_training function
        prepared_dataset = prepare_for_training(dataset, cache=True, batch_size=batch_size, shuffle_buffer_size=1000)

        # Check if the dataset is batched
        self.assertEqual(prepared_dataset._batch_size, batch_size)

        # Check if the dataset is mapped correctly
        for image, label in prepared_dataset:
            self.assertEqual(image.shape, (batch_size, 64, 64, 3))
            self.assertEqual(label.shape, (batch_size, 10))

    def test_getW(self):
        # Define the input arrays
        qClass = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        rClasses = np.array([[0, 0, 1], [1, 0, 0]])

        # Call the getW function
        W = getW(qClass, rClasses)

        # Check the output
        expected_output = np.array([0.9, 0.6])
        np.testing.assert_array_equal(W, expected_output)

    def test_getEuclidDist(self):
        # Define the input arrays
        qFeats = np.array([1, 1])
        rFeats = np.array([[2, 1], [1,1]])

        # Call the getEuclidDist function
        d = getEuclidDist(qFeats, rFeats)

        # Check the output
        expected_output = np.array([1, 0])
        np.testing.assert_allclose(d, expected_output)

    def test_getModel(self):
        # Call the getModel function
        model, checkpoint = getModel()

        # Check the output
        self.assertIsInstance(model, tf.keras.Model)
        self.assertIsInstance(checkpoint, tf.keras.callbacks.ModelCheckpoint)

if __name__ == '__main__':
    unittest.main()