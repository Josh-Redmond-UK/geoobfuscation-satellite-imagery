import unittest
import tensorflow as tf
from utils import GaussianDiffusion, AttentionBlock

class TestUtils(unittest.TestCase):


    def test_gaussian_diffusion(self):
        diffusion = GaussianDiffusion()

        # Test q_mean_variance method
        x_start = tf.random.normal(shape=(32, 64, 64, 3))
        t = 10
        mean, variance, log_variance = diffusion.q_mean_variance(x_start, t)
        self.assertEqual(mean.shape, (32, 1, 1, 1))
        self.assertEqual(variance.shape, (32, 1, 1, 1))
        self.assertEqual(log_variance.shape, (32, 1, 1, 1))

    def test_q_sample(self):
        diffusion = GaussianDiffusion()
        # Test q_mean_variance method
        x_start = tf.random.normal(shape=(32, 64, 64, 3))
        t = 10

        # Test q_sample method
        noise = tf.random.normal(shape=(32, 64, 64, 3))
        diffused_samples = diffusion.q_sample(x_start, t, noise)
        self.assertEqual(diffused_samples.shape, (32, 64, 64, 3))

    def test_predict_start_from_noise(self):
        # Create model to test
        diffusion = GaussianDiffusion()
        # Test q_mean_variance method
        x_start = tf.random.normal(shape=(32, 64, 64, 3))
        t = 10
        noise = tf.random.normal(shape=(32, 64, 64, 3))
        # Test predict_start_from_noise method
        x_t = tf.random.normal(shape=(32, 64, 64, 3))
        predicted_start = diffusion.predict_start_from_noise(x_t, t, noise)
        self.assertEqual(predicted_start.shape, (32, 64, 64, 3))



    def test_q_posterior(self):
        # Create model to test
        diffusion = GaussianDiffusion()
        # Test q_mean_variance method
        x_start = tf.random.normal(shape=(32, 64, 64, 3))
        t = 10
        x_t = tf.random.normal(shape=(32, 64, 64, 3))
        # Test q_posterior method
        posterior_mean, posterior_variance, posterior_log_variance = diffusion.q_posterior(x_start, x_t, t)
        self.assertEqual(posterior_mean.shape, (32, 64, 64, 3))
        self.assertEqual(posterior_variance.shape, (32, 64, 64, 3))
        self.assertEqual(posterior_log_variance.shape, (32, 64, 64, 3))

    def test_p_mean_variance(self):
        # Create model to test
        diffusion = GaussianDiffusion()
        # Test q_mean_variance method
        t = 10

        # Test p_mean_variance method
        pred_noise = tf.random.normal(shape=(32, 64, 64, 3))
        x = tf.random.normal(shape=(32, 64, 64, 3))
        model_mean, posterior_variance, posterior_log_variance = diffusion.p_mean_variance(pred_noise, x, t)
        self.assertEqual(model_mean.shape, (32, 64, 64, 3))
        self.assertEqual(posterior_variance.shape, (32, 64, 64, 3))
        self.assertEqual(posterior_log_variance.shape, (32, 64, 64, 3))

        # Test p_sample method
        sampled_images = diffusion.p_sample(pred_noise, x, t)
        self.assertEqual(sampled_images.shape, (32, 64, 64, 3))

    def test_attention_block(self):
        attention_block = AttentionBlock(units=64, groups=3)

        # Test call method
        inputs = tf.random.normal(shape=(32, 64, 64, 3))
        output = attention_block(inputs)
        self.assertEqual(output.shape, (32, 64, 64, 64))

if __name__ == '__main__':
    unittest.main()
