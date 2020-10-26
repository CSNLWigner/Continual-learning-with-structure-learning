import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np

def gamma_from_alpha(alpha):
  return np.array([tf.cos(tf.cast(alpha/180*np.pi,tf.float32)), tf.sin(tf.cast(alpha/180*np.pi,tf.float32))])

def generate_data(N=100, alpha=0, z_prior_type='uniform', sigma_z_prior=1, r_bias=0, sigma_reward=0.1, sigma_bias=0):
  gamma = gamma_from_alpha(alpha)

  if z_prior_type == 'normal':
    z_prior = tfd.MultivariateNormalDiag(loc=[0,0], scale_diag=[sigma_z_prior,sigma_z_prior]);
  elif z_prior_type == 'uniform':
    z_prior = tfd.Uniform([0,0],[sigma_z_prior,sigma_z_prior])

  z = np.array(z_prior.sample(N))

  r_noise = tfd.Normal(0, sigma_reward).sample(N)
  r_mean = tf.reduce_sum(tf.multiply(gamma,z),1) + r_bias
  r = r_mean + r_noise

  return {'z':z,'r':r}

# log likelihood of dataset for a model with fixed alpha
def model_log_llh(z, r, alpha, sigma_reward):
  gamma = gamma_from_alpha(alpha)
  prob_per_sample = tfp.distributions.Normal(loc=tf.reduce_sum(tf.multiply(gamma,z),1), scale=sigma_reward).log_prob(r)
  return tf.reduce_sum(prob_per_sample)

