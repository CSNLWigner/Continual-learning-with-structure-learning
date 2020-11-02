import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
        
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


def plot_data(data, labels=False):
    plt.scatter(*data['z'].T,c=data['r'])
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('z_1')
    plt.ylabel('z_2')
    if labels:
        labels = ['{0}'.format(i) for i in range(data['z'].shape[0])]
        for label, x, y in zip(labels, data['z'][:, 0], data['z'][:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))


def gamma_from_alpha(alpha):
    #return np.array([tf.cos(tf.cast(alpha/180*np.pi,tf.float32)), tf.sin(tf.cast(alpha/180*np.pi,tf.float32))])
    return np.array([np.cos(alpha/180*np.pi), np.sin(alpha/180*np.pi)]).astype(np.float32)


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


def model_log_llh_by_alpha(z, r, alpha, sigma_reward):
    gamma = gamma_from_alpha(alpha)
    return model_log_llh(z, r, gamma, sigma_reward, method='tf')


def model_log_llh(z, r, gamma, sigma_reward, method='tf'):
    if method == 'tf':
        prob_per_sample = tfp.distributions.Normal(loc=tf.reduce_sum(tf.multiply(gamma,z),1), scale=sigma_reward).log_prob(r)
        joint_prob = tf.reduce_sum(prob_per_sample)
    return joint_prob


def logsumexp(llhs):
    '''Computes the log of sum of probs in a numerically safe way. Needs list of log probs as input.'''
    llh_max = np.max(llhs)
    return np.log(np.sum(np.exp(llhs - llh_max))/len(llhs)) + llh_max


def compute_log_mllh(z, r, alpha_samples, sigma_reward):
    '''Computes mllh by integrating over samples from the prior given as arguments. Can emulate 1d model mllh by only supplying a single sample from gamma.'''
    log_llhs = np.array([model_log_llh_by_alpha(z, r, alpha=alpha_sample, sigma_reward=sigma_reward) for alpha_sample in alpha_samples])
    return logsumexp(log_llhs)


def compute_log_mllhs(z, r, list_of_list_of_alphas, sigma_reward, verbose=False):
    '''Computes mllhs for a list of models for a list of data points, given a list of gamma samples for each model (a list of lists)'''
    if verbose:
        pbar = tf.keras.utils.Progbar(len(z))
    mllhs = []
    for t in range(len(z)):
        mllhs.append([compute_log_mllh(z[:t+1],r[:t+1],alpha_samples,sigma_reward) for alpha_samples in list_of_list_of_alphas])
        if verbose:
            pbar.add(1)
    return mllhs


def index_of_model_change(mllhs, model_id=0, never_result=np.nan):
    '''Given a list of mllhs, computes first time index where best model is model_id'''
    ids_of_best_model = np.argmax(np.array(mllhs),1)
    if len(np.nonzero(ids_of_best_model == 0)[0]) == 0:
        id_change = never_result
    else:
        id_change = np.nonzero(ids_of_best_model == 0)[0][0]
    return id_change