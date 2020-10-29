import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np

def gamma_from_alpha(alpha):
    #return np.array([tf.cos(tf.cast(alpha/180*np.pi,tf.float32)), tf.sin(tf.cast(alpha/180*np.pi,tf.float32))])
    return np.array([np.cos(alpha/180*np.pi), np.sin(alpha/180*np.pi)]).astype(np.float32)
    
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


def model_llh_by_alpha(z, r, alpha, sigma_reward, method='tf'):
    gamma = gamma_from_alpha(alpha)
    return model_llh(z, r, gamma, sigma_reward, method='tf')

def model_llh(z, r, gamma, sigma_reward, method='tf'):
    if method == 'tf':
        prob_per_sample = tfp.distributions.Normal(loc=tf.reduce_sum(tf.multiply(gamma,z),1), scale=sigma_reward).prob(r)
        joint_prob = tf.reduce_prod(prob_per_sample)
    elif method == 'np':
        prob_per_sample = tf.exp(-0.5 * 
                (tf.reduce_sum(tf.multiply(gamma,z),1) - r)**2 / sigma_reward**2) / (2*np.pi*sigma_reward**2)**0.5
        joint_prob = tf.reduce_prod(prob_per_sample)
    #assert joint_prob != 0.0, 'joint probability too small, use log prob'
    return joint_prob


# log likelihood of dataset for a model with fixed alpha
def model_log_llh(z, r, alpha, sigma_reward):
    gamma = gamma_from_alpha(alpha)
    prob_per_sample = tf.exp(-0.5 * (tf.reduce_sum(tf.multiply(gamma,z),1) - r)**2 / sigma_reward**2)
    return tf.reduce_sum(prob_per_sample)
