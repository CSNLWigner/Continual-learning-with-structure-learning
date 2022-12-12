import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import zip_longest

########################################### Misc helper functions ###########################################

def riffle(list1,list2):
    '''
    Riffle indices for interleaved data.
    '''
    riffled = np.array(list(zip_longest(list1,list2))).flatten()
    riffled = [x for x in riffled if x is not None]
    return np.array(riffled)


def gamma_from_alpha(alpha):
    '''
    Convert the alpha parameter of a decision boundary to a gamma parameter.
    '''
    #return np.array([tf.cos(tf.cast(alpha/180*np.pi,tf.float32)), tf.sin(tf.cast(alpha/180*np.pi,tf.float32))])
    return np.array([np.cos(alpha/180*np.pi), np.sin(alpha/180*np.pi)]).astype(np.float32)


def index_of_model_change(mllhs, model_id=0, never_result=np.nan):
    '''Given a list of mllhs, computes first time index where best model is model_id'''
    ids_of_best_model = np.argmax(np.array(mllhs),1)
    if len(np.nonzero(ids_of_best_model == model_id)[0]) == 0:
        id_change = never_result
    else:
        id_change = np.nonzero(ids_of_best_model == model_id)[0][0]
    return id_change


def index_of_model_change_modified(mllhs, model_id=0, never_result=np.nan):
    '''Given a list of mllhs, computes first time index where best model is model_id'''
    ids_of_best_model = np.argmax(np.array(mllhs),1)
    if len(np.nonzero(ids_of_best_model == model_id)[0]) == 0:
        id_change = never_result
    else:
        a = ids_of_best_model == model_id
        if sum(a) == len(a):
            id_change = 0
        else:
            id_change = len(a) - np.argmin(np.flip(a))
    return id_change

def model_change_time(learning_dict, desired_model):
    try:
        switch_time = learning_dict['prominent_models'].index(desired_model)
    except ValueError:
        switch_time = np.nan
    return switch_time

########################################### Functions that take data dictionaries as input ###########################################

def concatenate_data(data1, data2):
    '''
    Join two data dictionaries.
    '''
    data_out = {}
    for key in data1.keys():
        data_out[key] = np.concatenate((data1[key], data2[key]))
    return data_out

def split_data(data, split):
    '''
    Split a data dictionary into two data dictionaries.
    '''
    data1 = {}
    data2 = {}
    for key in data.keys():
        data1[key] = data[key][:split]
        data2[key] = data[key][split:]
    return data1, data2

def split_data_by_index(data, indices):
    '''
    Split a data dictionary into two data dictionaries.
    '''
    data1 = {}
    data2 = {}
    for key in data.keys():
        data1[key] = data[key][indices]
        data2[key] = np.delete(data[key], indices, 0)
    return data1, data2

def reorder_data(data, indices):
    '''
    Reorder datapoints in a data dictionary by indices specified by a numpy array.
    '''
    data_out = {}
    for key in data.keys():
        data_out[key] = data[key][indices]
    return data_out

def plot_data(data, labels=False, limit=1.75, climit=1, show_axes=True, axislabels=True, colorbar=True, ticks=True, marker='o', figsize=None):
    '''
    Create a plot from a data dictionary.
    '''
    plt.scatter(*data['z'].T,c=data['r'], marker=marker, vmin=-climit, vmax=climit)
    plt.gca().set_aspect('equal')
    plt.xlim([-limit,limit])
    plt.ylim([-limit,limit])
    if show_axes:
        plt.axvline(x = 0, color = 'k', linestyle = '--')
        plt.axhline(y = 0, color = 'k', linestyle = '--')
    if axislabels:
        plt.xlabel('z_1')
        plt.ylabel('z_2')
    if colorbar:
        plt.colorbar()
    if not ticks:
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False) # labels along the bottom edge are off
    if labels:
        labels = ['{0}'.format(i) for i in range(data['z'].shape[0])]
        for label, x, y in zip(labels, data['z'][:, 0], data['z'][:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    if figsize is not None:
        plt.gcf().set_size_inches(figsize)

def plot_data_subplots(data_list, labels=False, limit=1.75, climit=1, show_axes=True, axislabels=True, ticks=True, marker='o', figsize=None, titles=None):
    '''
    Create a plot from a data dictionary.
    '''
    for i in range(len(data_list)):
        plt.subplot(1, len(data_list), i + 1)
        plt.scatter(*data_list[i]['z'].T,c=data_list[i]['r'], marker=marker, vmin=-climit, vmax=climit)
        plt.gca().set_aspect('equal')
        plt.xlim([-limit,limit])
        plt.ylim([-limit,limit])
        if show_axes:
            plt.axvline(x = 0, color = 'k', linestyle = '--')
            plt.axhline(y = 0, color = 'k', linestyle = '--')
        if titles is not None:
            plt.title(titles[i])
        if i==0:
            if axislabels:
                plt.xlabel('z_1')
                plt.ylabel('z_2')
            if labels:
                labels = ['{0}'.format(i) for i in range(data_list[i]['z'].shape[0])]
                for label, x, y in zip(labels, data_list[i]['z'][:, 0], data_list[i]['z'][:, 1]):
                    plt.annotate(
                        label,
                        xy=(x, y), xytext=(-20, 20),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        if not ticks:
            plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            left=False,
            labelleft=False) # labels along the bottom edge are off
    if figsize is not None:
        plt.gcf().set_size_inches(figsize)


def plot_mmllh_curves(learning_dict, model_set, T, color_dict, figsize=None, indicate_best_model = True):
    '''
    Plot the mean mmllh curves for a set of models. Option to indicate the best model at each time step.
    '''
    plt.figure()
    x = np.arange(1, T + 1)
    for model in model_set:
        mmllh = learning_dict[model]['mmllh']
        color = color_dict['model_' + model]
        plt.plot(x, np.mean(np.log(mmllh), axis = 0), label = model, linewidth = 4, color = color)
    if indicate_best_model:
        for t in range(T):
            mmllh_t = np.array([learning_dict[model]['mmllh'][:,t] for model in model_set])
            best_model = model_set[np.argmax(np.mean(mmllh_t, axis = 1))]
            color = color_dict['model_' + best_model]
            plt.plot(t + 1, 0, 'o', color = color, markersize = 10)
    plt.xlabel('time')
    plt.ylabel('log mmllh')
    plt.legend()
    if figsize is not None:
        plt.gcf().set_size_inches(figsize)


def plot_mllh_curves_subpanels(learning_dicts, model_set, T, color_dict, figsize=None, indicate_best_model = True, titles=None):
    '''
    Plot mean mllh curves for a set of models in subplots for different learners.
    Input is a list of result dictionaries. 
    Option to indicate the best model at each time step.
    '''
    plt.figure()
    x = np.arange(1, T + 1)
    for i, learning_dict in enumerate(learning_dicts):
        plt.subplot(1, len(learning_dicts), i + 1)
        for model in model_set:
            mllh = learning_dict[model]['mmllh']
            color = color_dict['model_' + model]
            plt.plot(x, np.mean(np.log(mllh), axis = 0), label = model, linewidth = 4, color = color)
        if indicate_best_model:
            for t in range(T):
                mllh_t = np.array([learning_dict[model]['mmllh'][:,t] for model in model_set])
                best_model = model_set[np.argmax(np.mean(mllh_t, axis = 1))]
                color = color_dict['model_' + best_model]
                plt.plot(t + 1, 0, 'o', color = color, markersize = 10)
        if titles is not None:
            plt.title(titles[i])
        #Labels
        if i == 0:
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('log mllh')
    if figsize is not None:
        plt.gcf().set_size_inches(figsize)

########################################### Functions associated with 1x2D model ###########################################


def generate_data_from_gamma(N=100, gamma=np.array([0,1]), z_prior_type='uniform', sigma_z_prior=1, r_bias=0, sigma_reward=0.1, sigma_bias=0, context_value=0):
    '''Generate data from an 1x2D model'''
    if z_prior_type == 'normal':
        z_prior = tfd.MultivariateNormalDiag(loc=[0,0], scale_diag=[sigma_z_prior,sigma_z_prior]);
    elif z_prior_type == 'uniform':
        z_prior = tfd.Uniform([-sigma_z_prior,-sigma_z_prior],[sigma_z_prior,sigma_z_prior])
    elif z_prior_type == 'informative':
        z_prior = tfd.MultivariateNormalDiag(loc=[-1,1], scale_diag=[.3,.3]);
    z = np.array(z_prior.sample(N))

    r_noise = tfd.Normal(0, sigma_reward).sample(N)
    r_mean = tf.cast(tf.reduce_sum(tf.multiply(gamma,z),1),dtype=tf.float32) + r_bias
    r = r_mean + r_noise
    
    c = np.ones(N) * context_value

    b = c

    return {'z':z, 'r':np.array(r), 'c':c, 'b':b}

        
def generate_data(N=100, alpha=0, z_prior_type='uniform', sigma_z_prior=1, r_bias=0, sigma_reward=0.1, sigma_bias=0, context_value=0):
    '''Generate data from 1x2D model by alpha'''
    gamma = gamma_from_alpha(alpha)
    return generate_data_from_gamma(N=N, gamma=gamma, z_prior_type=z_prior_type,
                                    sigma_z_prior=sigma_z_prior, r_bias=r_bias,
                                    sigma_reward=sigma_reward, sigma_bias=sigma_bias, context_value=context_value)


def gamma_posterior_analytic(zs, rs, sigma_r, Sigma_0):
    '''
    Calculate the parameters of the posterior of an 1x2D model.
    Output: mu_T and Sigma_T of the posterior p(\gamma | D) = N(\gamma; mu_T, Sigma_T)
    Notation as in the 'CL marginal likelihood 2' calculation in onenote
    The core derivation comes from the function 'trial_nonorm_posterior_set_transformed'
    '''
    T = np.size(zs,0)
    detSigma_0 = np.linalg.det(Sigma_0)
    Sigma_i_star_invs = []
    Sigma_i_invs = []
    mu_is = []
    for t in range(T):
        z = zs[t]
        r = rs[t]
        Sigma_i_star_inv = np.array([[z[0]**2/sigma_r**2, z[0]*z[1]/sigma_r**2],[z[0]*z[1]/sigma_r**2, z[1]**2/sigma_r**2]])
        Sigma_i_star_invs.append(Sigma_i_star_inv)
        if t==0:
            Sigma_i_inv = Sigma_i_star_inv + np.linalg.inv(Sigma_0)
        else:
            Sigma_i_inv = Sigma_i_star_inv + Sigma_i_invs[t-1]
        Sigma_i_invs.append(Sigma_i_inv)
        Sigma_i = np.linalg.inv(Sigma_i_inv)
        if t==0:
            mu_i = Sigma_i.dot(z*r/sigma_r**2)
        else:
            mu_i = Sigma_i.dot(z*r/sigma_r**2 + Sigma_i_invs[t-1].dot(mu_is[t-1]) )
        mu_is.append(mu_i)
    mu_T = mu_i
    Sigma_T = Sigma_i
    return mu_T, Sigma_T


def model_marginal_llh_analytic(zs, rs, sigma_r, Sigma_0):
    '''
    this is the model marginal likelihood function (1x2D model?)
    it is validated through 'trial_nonorm_posterior_set_transformed'
    from that function the only step fowrad is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    '''
    T = np.size(zs,0)
    detSigma_0 = np.linalg.det(Sigma_0)
    Sigma_i_star_invs = []
    Sigma_i_invs = []
    mu_is = []
    y = 1/(2*np.pi)/np.sqrt(np.linalg.det(Sigma_0))
    for t in range(T):
        z = zs[t]
        r = rs[t]
        Sigma_i_star_inv = np.array([[z[0]**2/sigma_r**2, z[0]*z[1]/sigma_r**2],[z[0]*z[1]/sigma_r**2, z[1]**2/sigma_r**2]])
        Sigma_i_star_invs.append(Sigma_i_star_inv)
        if t==0:
            Sigma_i_inv = Sigma_i_star_inv + np.linalg.inv(Sigma_0)
        else:
            Sigma_i_inv = Sigma_i_star_inv + Sigma_i_invs[t-1]
        Sigma_i_invs.append(Sigma_i_inv)
        Sigma_i = np.linalg.inv(Sigma_i_inv)
        if t==0:
            mu_i = Sigma_i.dot(z*r/sigma_r**2)
        else:
            mu_i = Sigma_i.dot(z*r/sigma_r**2 + Sigma_i_invs[t-1].dot(mu_is[t-1]) )
        mu_is.append(mu_i)
        y = y * multivariate_normal.pdf(r, mean = 0, cov = sigma_r**2)
    y = y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i)
    return y

########################################### ? ###########################################


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


def compute_log_mllh_by_gamma(z, r, gamma_samples, sigma_reward):
    '''Computes mllh by integrating over samples from the prior given as arguments. Can emulate 1d model mllh by only supplying a single sample from gamma.'''
    log_llhs = np.array([model_log_llh(z, r, gamma=gamma_sample, sigma_reward=sigma_reward) for gamma_sample in gamma_samples])
    return logsumexp(log_llhs)


def compute_log_mllhs_by_gamma(z, r, list_of_list_of_gammas, sigma_reward, verbose=False):
    '''Computes mllhs for a list of models for a list of data points, given a list of gamma samples for each model (a list of lists)'''
    if verbose:
        pbar = tf.keras.utils.Progbar(len(z))
    mllhs = []
    for t in range(len(z)):
        mllhs.append([compute_log_mllh_by_gamma(z[:t+1],r[:t+1],gamma_samples,sigma_reward) for gamma_samples in list_of_list_of_gammas])
        if verbose:
            pbar.add(1)
    return mllhs


########################################### Legacy code ###########################################

from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def model_marginal_llh_analytic_new(zs, rs, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), model = '2d'):
    '''
    Analytic computation of marginal likelihood of ? model
    it is validated through 'trial_nonorm_posterior_set_transformed'
    from that function the only step fowrad is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    '''
    if zs.size != 0:
        T = np.size(zs,0)
        if model == '2d':
            assert not np.isscalar(Sigma_0), 'Sigma_0 must be a 2-dimensional array'
            detSigma_0 = np.linalg.det(Sigma_0)
            Sigma_i_star_invs = []
            Sigma_i_invs = []
            mu_is = []
            y = 1/(2*np.pi)/np.sqrt(np.linalg.det(Sigma_0))
            for t in range(T):
                z = zs[t]
                r = rs[t]
                Sigma_i_star_inv = np.array([[z[0]**2/sigma_r**2, z[0]*z[1]/sigma_r**2],[z[0]*z[1]/sigma_r**2, z[1]**2/sigma_r**2]])
                Sigma_i_star_invs.append(Sigma_i_star_inv)
                if t==0:
                    Sigma_i_inv = Sigma_i_star_inv + np.linalg.inv(Sigma_0)
                else:
                    Sigma_i_inv = Sigma_i_star_inv + Sigma_i_invs[t-1]
                Sigma_i_invs.append(Sigma_i_inv)
                Sigma_i = np.linalg.inv(Sigma_i_inv)
                if t==0:
                    mu_i = Sigma_i.dot(z*r/sigma_r**2)
                else:
                    mu_i = Sigma_i.dot(z*r/sigma_r**2 + Sigma_i_invs[t-1].dot(mu_is[t-1]))
                mu_is.append(mu_i)
                y = y * multivariate_normal.pdf(r, mean = 0, cov = sigma_r**2)
            y = y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i)
        else:
            '''
            Sigma_0 is the standard deviation of the gamma prior
            '''
            assert np.isscalar(Sigma_0), 'Sigma_0 must be scalar'
            if model == 'x':
                integral_dim = 1
            else:
                integral_dim = 0

            Sigma_i_star_invs = []
            Sigma_i_invs = []
            mu_is = []
            y = 1/(np.sqrt(2*np.pi))/Sigma_0
            for t in range(T):
                z = zs[t]
                r = rs[t]
            
                Sigma_i_star_inv = z[integral_dim]**2/sigma_r**2
                Sigma_i_star_invs.append(Sigma_i_star_inv)
                if t==0:
                        Sigma_i_inv = Sigma_i_star_inv + 1/Sigma_0**2
                else:
                        Sigma_i_inv = Sigma_i_star_inv + Sigma_i_invs[t-1]
                Sigma_i_invs.append(Sigma_i_inv)
                Sigma_i = 1/Sigma_i_inv
                if t==0:
                        mu_i = Sigma_i * z[integral_dim]*r/sigma_r**2
                else:
                        mu_i = Sigma_i * (z[integral_dim]*r/sigma_r**2 + Sigma_i_invs[t-1]*mu_is[t-1])
                mu_is.append(mu_i)
                y = y * multivariate_normal.pdf(r, mean = 0, cov = sigma_r**2)
            y = y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i)
        return y
    else:
        return 1.


def model_marginal_llh_analytic_2x2D(z, r, sigma_r, Sigma_0_2D = np.array([[1., 0.], [0., 1.]]), verbose = True):
    '''
    Analytic computation of marginal likelihood of 2x2D model
    it is validated through 'trial_nonorm_posterior_set_transformed'
    from that function the only step forward is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    '''
    T = z.shape[0]
    
    indices = np.arange(T)
    index_subsets = list(powerset(indices))

    mmllh_accumulator = 0.
    if verbose:
        pbar = tf.keras.utils.Progbar(len(index_subsets))
    for index_subset in index_subsets:
        z1 = z[list(index_subset)]
        r1 = r[list(index_subset)]
        
        complementer_subset = [item for item in indices if item not in index_subset]
        
        z2 = z[complementer_subset]
        r2 = r[complementer_subset]
        
        mmllh_accumulator += model_marginal_llh_analytic_new(z1, r1, sigma_r, Sigma_0 = Sigma_0_2D, model = '2d') \
        * model_marginal_llh_analytic_new(z2, r2, sigma_r, Sigma_0 = Sigma_0_2D, model = '2d')
        
        if verbose:
            pbar.add(1)
            
    mmllh_accumulator /= 2**T
    return mmllh_accumulator