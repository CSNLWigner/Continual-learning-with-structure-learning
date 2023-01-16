import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.stats import multivariate_normal
from copy import deepcopy
import pandas as pd


def generate_batch_data_old(task_alphas, batch_size, num_of_batches):
  data1 = generate_data(batch_size, alpha = task_alphas[0])
  data2 = generate_data(batch_size, alpha = task_alphas[1])
  data = concatenate_data(data1, data2)
  for i in range(1, num_of_batches):
    data1 = generate_data(batch_size, alpha = task_alphas[0])
    data2 = generate_data(batch_size, alpha = task_alphas[1])
    data_ = concatenate_data(data1, data2)
    data = concatenate_data(data, data_)
  return data

def generate_batch_data(task_alphas, batch_size, num_of_batches):
  batches = []
  for i in range(num_of_batches):
    batch = concatenate_data([generate_data(batch_size, alpha = alpha) for alpha in task_alphas])
    batches.append(batch)
  return concatenate_data(batches)

  
def plot_mmllh_curves(learning_dict, model_set, D, T, color_dict, filepath, title):
  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib.pyplot import rcParams
  rcParams['font.size'] = 15
  rcParams['figure.figsize'] = (15, 10)
  plt.figure()
  x = np.arange(1, T + 1)
  for model in model_set:
    mmllh = learning_dict[model]['mmllh']
    color = color_dict['model_' + model]
    plt.plot(x, np.mean(np.log(mmllh), axis = 0), label = model, linewidth = 4, color = color)
  plt.xlabel('number of points')
  plt.ylabel('log mmllh')
  plt.legend()
  plt.title(title)
  plt.savefig(filepath)

def infer_angles_from_gammas(gammas):
  return list(np.arctan(gammas[:, 1] / gammas[:, 0]) * 180 / np.pi)

def normalize_weights(weights):
  norm = sum(weights)
  weights = np.array(weights)/norm
  return list(weights), norm

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_data(N=100, alpha=0, z_prior_type='normal', sigma_z_prior=1, r_bias=0, sigma_reward=0.001, sigma_bias=0):
    gamma = gamma_from_alpha(alpha)

    if z_prior_type == 'normal':
        z_prior = tfd.MultivariateNormalDiag(loc=[0,0], scale_diag=[sigma_z_prior,sigma_z_prior]);
    elif z_prior_type == 'uniform':
        z_prior = tfd.Uniform([-sigma_z_prior,-sigma_z_prior],[sigma_z_prior,sigma_z_prior])

    z = np.array(z_prior.sample(N))
    c = [str(alpha)] * N
    r_noise = tfd.Normal(0, sigma_reward).sample(N)
    r_mean = tf.reduce_sum(tf.multiply(gamma,z),1) + r_bias
    r = r_mean + r_noise

    return {'z':z,'r':r, 'c':c}


def plot_data(data, labels=False):
    plt.scatter(*data['z'].T,c=data['r'])
    plt.gca().set_aspect('equal')
    plt.colorbar()
    #plt.xlim([0,1])
    #plt.ylim([0,1])
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

def index_of_first_model_change(mllhs, model_id=0, never_result=np.nan):
    '''Given a list of mllhs, computes first time index where best model is model_id'''
    ids_of_best_model = np.argmax(np.array(mllhs),1)
    if len(np.nonzero(ids_of_best_model == model_id)[0]) == 0:
        id_change = never_result
    else:
        a = ids_of_best_model == model_id
        if sum(a) == len(a):
            id_change = 0
        else:
            id_change = len(a) - np.argmin(a)
    return id_change


def concatenate_data_old(data1, data2):
  z = np.concatenate((data1['z'], data2['z']), 0)
  r = np.concatenate((np.array(data1['r']), np.array(data2['r'])))
  c = data1['c'] + data2['c']
  return {'z': z, 'r': r, 'c': c}


def concatenate_data(data_iterable):
  z = np.concatenate([item['z'] for item in data_iterable], 0)
  r = np.concatenate([np.array(item['r']) for item in data_iterable])
  c = np.concatenate([np.array(item['c']) for item in data_iterable])
  return {'z': z, 'r': r, 'c': list(c)}


def gamma_posterior_analytic(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), model = '2d'):
    zs = data['z']
    rs = np.array(data['r'])
    if zs.size != 0:
      T = np.size(zs,0)
      if model == '2d':
        assert not np.isscalar(Sigma_0), 'Sigma_0 must be a 2-dimensional array'
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
                mu_i = Sigma_i.dot(z*r/sigma_r**2 + Sigma_i_invs[t-1].dot(mu_is[t-1]))
            mu_is.append(mu_i)
            
        mu_T = mu_i
        Sigma_T = Sigma_i
        return mu_T, Sigma_T
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
            
        mu_T = mu_i
        Sigma_T = Sigma_i
        return mu_T, Sigma_T
    else:
      if model != '2d':
        return 0., Sigma_0
      else:
        return np.array([0., 0.]), Sigma_0

def model_marginal_llh_analytic_list(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), model = '2d'):
    # this is the model marginal likelihood function
    # it is validated through 'trial_nonorm_posterior_set_transformed'
    # from that function the only step fowrad is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    zs = data['z']
    rs = np.array(data['r'])
    if zs.size != 0:
      T = np.size(zs,0)
      if model == '2d':
        assert not np.isscalar(Sigma_0), 'Sigma_0 must be a 2-dimensional array'
        detSigma_0 = np.linalg.det(Sigma_0)
        Sigma_i_star_invs = []
        Sigma_i_invs = []
        mu_is = []
        y = 1/(2*np.pi)/np.sqrt(np.linalg.det(Sigma_0))
        ys = []
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
            ys.append(y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i))
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
        ys = []
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
            ys.append(y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i))

      return ys
    else:
      return 1.


def model_marginal_llh_analytic(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), model = '2d'):
    # this is the model marginal likelihood function
    # it is validated through 'trial_nonorm_posterior_set_transformed'
    # from that function the only step fowrad is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    zs = data['z']
    rs = np.array(data['r'])
    if zs.size != 0:
      T = np.size(zs,0)
      if model == '2d':
        assert not np.isscalar(Sigma_0), 'Sigma_0 must be a 2-dimensional array'
        detSigma_0 = np.linalg.det(Sigma_0)
        Sigma_i_star_invs = []
        Sigma_i_invs = []
        mu_is = []
        y = 1/(2*np.pi)/np.sqrt(np.linalg.det(Sigma_0))
        ys = []
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
            ys.append(y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i))
        return y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i)
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
        ys = []
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
            ys.append(y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i))

        return y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i)
    else:
      return 1.

def model_marginal_llh_analytic_2x2D(data, sigma_r, Sigma_0_2D = np.array([[1., 0.], [0., 1.]]), verbose = True):
  z = data['z']
  r = np.array(data['r'])
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
    
    mmllh_accumulator += model_marginal_llh_analytic(z1, r1, sigma_r, Sigma_0 = Sigma_0_2D, model = '2d') \
    * model_marginal_llh_analytic(z2, r2, sigma_r, Sigma_0 = Sigma_0_2D, model = '2d')
    
    if verbose:
      pbar.add(1)
      
  mmllh_accumulator /= 2**T
  return mmllh_accumulator

def model_marginal_llh_analytic_2x2D_opposite_dir(data, sigma_r, Sigma_0_2D = np.array([[1., 0.], [0., 1.]]), verbose = True):
  z = data['z']
  r = np.array(data['r'])
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
    mu_1, sigma_1 = gamma_posterior_analytic(z1, r1, sigma_r)
    mu_2, sigma_2 = gamma_posterior_analytic(z2, r2, sigma_r)

    mmllh_accumulator += model_marginal_llh_analytic(z1, r1, sigma_r, Sigma_0 = Sigma_0_2D, model = '2d') \
    * model_marginal_llh_analytic(z2, r2, sigma_r, Sigma_0 = Sigma_0_2D, model = '2d') \
    * multivariate_normal.pdf(mu_2, mean = -mu_1, cov = sigma_1 + sigma_2)
    
    if verbose:
      pbar.add(1)
      
  mmllh_accumulator /= 2**T
  return mmllh_accumulator

def normalize_weights(weights):
  norm = sum(weights)
  weights = np.array(weights)/norm
  return list(weights), norm

def model_marginal_llh_analytic_2x2D_PF_randomized_list(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), N = 64, N2 = 64, verbose = True):
  
  global frequencies
  global kell_pruning
  global chosen_weights
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  normalized_weights = [1.]
  mus = [[np.array([0., 0.]), np.array([0., 0.])]]
  Sigmas = [[Sigma_0, Sigma_0]]
  mmllh = 1.
  mmllhs = []
  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listabanS

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  for t in range(1, T + 1): # t darabszam
    
    # muk, sigmak, weightek a prune-olas elott:
    # normalizalatlanul megy ki before
    weights_before = deepcopy(normalized_weights)
    mus_before = deepcopy(mus)
    sigmas_before = deepcopy(Sigmas)

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 10
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind = N2
      while ind != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind]))
        ind += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind])
      value_counts = samples_from_categ_dist.value_counts()
      chosen_weights = dict()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      
      for indd in range(len(indices)):
        key = str(normalized_weights[indices[indd]])
        if key in list(chosen_weights.keys()):
          chosen_weights[key + '_'] = frequencies[indd]
        else:
          chosen_weights[key] = frequencies[indd]

      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])

  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    

    zt = z[t-1]
    rt = r[t-1]
    Sigma_star_inv = np.array([[zt[0]**2/sigma_r**2, zt[0]*zt[1]/sigma_r**2],[zt[0]*zt[1]/sigma_r**2, zt[1]**2/sigma_r**2]])
    
    for k in range(len(mus)):
   
      #sulymodositas
      if k <= int(len(mus)/2) - 1: # ekkor a task1-hez tartozo mu, sigma es weight frissul (task1-hez soroljuk az uj adatpontot)
        sigma_inv = np.linalg.inv(Sigmas[k][0])
        Sigma_1 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_1 = np.matmul(Sigma_1, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][0]))

        c_1 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][0], cov = Sigmas[k][0]) / \
        multivariate_normal.pdf(mu_1, mean = [0., 0.], cov = Sigma_1)

        normalized_weights[k] *= c_1
        mus[k][0] = mu_1
        Sigmas[k][0] = Sigma_1
       
        
      else: # ekkor a task2-hoz tartozo mu, sigma es weight frissul (task2-hoz soroljuk az uj adatpontot)
        sigma_inv = np.linalg.inv(Sigmas[k][1])
        Sigma_2 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_2 = np.matmul(Sigma_2, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][1]))


        c_2 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][1], cov = Sigmas[k][1]) / \
        multivariate_normal.pdf(mu_2, mean = [0., 0.], cov = Sigma_2)

        normalized_weights[k] *= c_2
        mus[k][1] = mu_2
        Sigmas[k][1] = Sigma_2
    
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)

    mmllhs.append(mmllh)
    if verbose:
      pbar.add(1)
   
    # t adatpont eseten posterior plotolasa pruning elott/utan:
    #plot_posteriors_before_after_pruning(t, mus_before, sigmas_before, mus, Sigmas, weights_before, normalized_weights)
    
    
  return mmllhs

def model_marginal_llh_analytic_2x2D_PF_randomized(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), N = 64, N2 = 64, verbose = True):
  
  global frequencies
  global kell_pruning
  global chosen_weights
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  normalized_weights = [1.]
  mus = [[np.array([0., 0.]), np.array([0., 0.])]]
  Sigmas = [[Sigma_0, Sigma_0]]
  mmllh = 1.
  mmllhs = []
  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listabanS

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  for t in range(1, T + 1): # t darabszam
    
    # muk, sigmak, weightek a prune-olas elott:
    # normalizalatlanul megy ki before
    weights_before = deepcopy(normalized_weights)
    mus_before = deepcopy(mus)
    sigmas_before = deepcopy(Sigmas)

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 10
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind = N2
      while ind != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind]))
        ind += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind])
      value_counts = samples_from_categ_dist.value_counts()
      chosen_weights = dict()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      
      for indd in range(len(indices)):
        key = str(normalized_weights[indices[indd]])
        if key in list(chosen_weights.keys()):
          chosen_weights[key + '_'] = frequencies[indd]
        else:
          chosen_weights[key] = frequencies[indd]

      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])

  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    

    zt = z[t-1]
    rt = r[t-1]
    Sigma_star_inv = np.array([[zt[0]**2/sigma_r**2, zt[0]*zt[1]/sigma_r**2],[zt[0]*zt[1]/sigma_r**2, zt[1]**2/sigma_r**2]])
    
    for k in range(len(mus)):
   
      #sulymodositas
      if k <= int(len(mus)/2) - 1: # ekkor a task1-hez tartozo mu, sigma es weight frissul (task1-hez soroljuk az uj adatpontot)
        sigma_inv = np.linalg.inv(Sigmas[k][0])
        Sigma_1 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_1 = np.matmul(Sigma_1, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][0]))

        c_1 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][0], cov = Sigmas[k][0]) / \
        multivariate_normal.pdf(mu_1, mean = [0., 0.], cov = Sigma_1)

        normalized_weights[k] *= c_1
        mus[k][0] = mu_1
        Sigmas[k][0] = Sigma_1
       
        
      else: # ekkor a task2-hoz tartozo mu, sigma es weight frissul (task2-hoz soroljuk az uj adatpontot)
        sigma_inv = np.linalg.inv(Sigmas[k][1])
        Sigma_2 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_2 = np.matmul(Sigma_2, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][1]))


        c_2 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][1], cov = Sigmas[k][1]) / \
        multivariate_normal.pdf(mu_2, mean = [0., 0.], cov = Sigma_2)

        normalized_weights[k] *= c_2
        mus[k][1] = mu_2
        Sigmas[k][1] = Sigma_2
    
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)

    mmllhs.append(mmllh)
    if verbose:
      pbar.add(1)
   
    # t adatpont eseten posterior plotolasa pruning elott/utan:
    #plot_posteriors_before_after_pruning(t, mus_before, sigmas_before, mus, Sigmas, weights_before, normalized_weights)
    
    
  return mmllh


def model_marginal_llh_analytic_2x2D_PF_randomized_opp_dir(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), N = 64, N2 = 64, verbose = True):

  global frequencies
  global kell_pruning
  global chosen_weights
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  normalized_weights = [1.]
  normalized_weights_integral = [1.]
  mus = [[np.array([0., 0.]), np.array([0., 0.])]]
  Sigmas = [[Sigma_0, Sigma_0]]
  mmllh = 1.
  mmllh_integral = 1.
  mmllhs = []
  row = 0
  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listaban

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  for t in range(1, T + 1): # t darabszam
    
    # muk, sigmak, weightek a prune-olas elott:
    # normalizalatlanul megy ki before
    weights_before = deepcopy(normalized_weights)
    mus_before = deepcopy(mus)
    sigmas_before = deepcopy(Sigmas)

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 10
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind = N2
      while ind != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind]))
        ind += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind])
      value_counts = samples_from_categ_dist.value_counts()
      chosen_weights = dict()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      
      for indd in range(len(indices)):
        key = str(normalized_weights[indices[indd]])
        if key in list(chosen_weights.keys()):
          chosen_weights[key + '_'] = frequencies[indd]
        else:
          chosen_weights[key] = frequencies[indd]

      normalized_weights, _ = normalize_weights(list(frequencies))
      
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])

  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    normalized_weights_integral = deepcopy(normalized_weights_integral) + deepcopy(normalized_weights_integral)
    

    zt = z[t-1]
    rt = r[t-1]
    Sigma_star_inv = np.array([[zt[0]**2/sigma_r**2, zt[0]*zt[1]/sigma_r**2],[zt[0]*zt[1]/sigma_r**2, zt[1]**2/sigma_r**2]])
    
    for k in range(len(mus)):
   
      #sulymodositas
      if k <= int(len(mus)/2) - 1: # ekkor a task1-hez tartozo mu, sigma es weight frissul (task1-hez soroljuk az uj adatpontot)
        sigma_inv = np.linalg.inv(Sigmas[k][0])
        Sigma_1 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_1 = np.matmul(Sigma_1, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][0]))
        
        c_1 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][0], cov = Sigmas[k][0]) / \
        multivariate_normal.pdf(mu_1, mean = [0., 0.], cov = Sigma_1)
        

        #if t != T:
        #  normalized_weights[k] *= c_1
        #else:
        #  normalized_weights[k] *= c_1 * multivariate_normal.pdf(mu_1, mean = -mus[k][1], cov = Sigma_1 + Sigmas[k][1])

        normalized_weights[k] *= c_1
        normalized_weights_integral[k] = normalized_weights[k] * multivariate_normal.pdf(mu_1, mean = -mus[k][1], cov = Sigma_1 + Sigmas[k][1])
        mus[k][0] = mu_1
        Sigmas[k][0] = Sigma_1
        
        
      else: # ekkor a task2-hoz tartozo mu, sigma es weight frissul (task2-hoz soroljuk az uj adatpontot)
        sigma_inv = np.linalg.inv(Sigmas[k][1])
        Sigma_2 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_2 = np.matmul(Sigma_2, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][1]))
        
      
        c_2 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][1], cov = Sigmas[k][1]) / \
        multivariate_normal.pdf(mu_2, mean = [0., 0.], cov = Sigma_2)
        
        
        #if t != T:
        #  normalized_weights[k] *= c_2
        #else:
        #  normalized_weights[k] *= c_2 * multivariate_normal.pdf(mus[k][0], mean = -mu_2, cov = Sigmas[k][0] + Sigma_2)

        normalized_weights[k] *= c_2
        normalized_weights_integral[k] = normalized_weights[k] * multivariate_normal.pdf(mus[k][0], mean = -mu_2, cov = Sigmas[k][0] + Sigma_2)
        mus[k][1] = mu_2
        Sigmas[k][1] = Sigma_2
    
    
    if kell_pruning:
      mmllh_save = deepcopy(mmllh)
      norm = sum(normalized_weights)
      mmllh *= (norm / 2)
      norm_integral = sum(normalized_weights_integral)
      mmllh_integral = mmllh_save * (norm_integral / 2)
    else:
      mmllh_save = deepcopy(mmllh)
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm / 2)
      norm_integral = sum(normalized_weights_integral)
      mmllh_integral = mmllh_save * (norm_integral / 2)

    mmllhs.append(mmllh_integral)
    if verbose:
      pbar.add(1)
   
    # t adatpont eseten posterior plotolasa pruning elott/utan:
    #plot_posteriors_before_after_pruning(t, mus_before, sigmas_before, mus, Sigmas, weights_before, normalized_weights)
    
    
  return mmllhs

def model_marginal_llh_analytic_2x1D_PF_randomized_list(data, sigma_r, Sigma_0 = 1., N = 64, N2 = 64, verbose = True):
  
  global frequencies
  global kell_pruning
  global chosen_weights
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  normalized_weights = [1.]
  mus = [[0., 0.]]
  Sigmas = [[Sigma_0, Sigma_0]]
  mmllh = 1.
  mmllhs = []
  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listaban

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)

  for t in range(1, T + 1): # t darabszam
    
    # muk, sigmak, weightek a prune-olas elott:
    # normalizalatlanul megy ki before
    weights_before = deepcopy(normalized_weights)
    mus_before = deepcopy(mus)
    sigmas_before = deepcopy(Sigmas)

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 10
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind = N2
      while ind != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind]))
        ind += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind])
      value_counts = samples_from_categ_dist.value_counts()
      chosen_weights = dict()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      for indd in range(len(indices)):
        key = str(normalized_weights[indices[indd]])
        if key in list(chosen_weights.keys()):
          chosen_weights[key + '_'] = frequencies[indd]
        else:
          chosen_weights[key] = frequencies[indd]

      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])

  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    

    zt = z[t-1]
    rt = r[t-1]
    
    for k in range(len(mus)):
      
      #sulymodositas
      if k <= int(len(mus)/2) - 1: # xhez soroljuk az ujat
        Sigma_star_inv = zt[1]**2/sigma_r**2
        sigma_inv = 1/Sigmas[k][0]
        Sigma_1 = 1/(Sigma_star_inv + sigma_inv)
        mu_1 = Sigma_1 * (zt[1]*rt/sigma_r**2 + sigma_inv * mus[k][0])

        c_1 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf(0., mean = mus[k][0], cov = Sigmas[k][0]) / \
        multivariate_normal.pdf(mu_1, mean = 0., cov = Sigma_1)

        normalized_weights[k] *= c_1
        mus[k][0] = mu_1
        Sigmas[k][0] = Sigma_1
       
        
      else: # yhoz soroljuk az ujat
        Sigma_star_inv = zt[0]**2/sigma_r**2
        sigma_inv = 1/Sigmas[k][1]
        Sigma_2 = 1/(Sigma_star_inv + sigma_inv)
        mu_2 = Sigma_2 * (zt[0]*rt/sigma_r**2 + sigma_inv * mus[k][1])

        c_2 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf(0., mean = mus[k][1], cov = Sigmas[k][1]) / \
        multivariate_normal.pdf(mu_2, mean = 0., cov = Sigma_2)

        normalized_weights[k] *= c_2
        mus[k][1] = mu_2
        Sigmas[k][1] = Sigma_2
    
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)

    mmllhs.append(mmllh)
    if verbose:
      pbar.add(1)
   
    # t adatpont eseten posterior plotolasa pruning elott/utan:
    #plot_posteriors_before_after_pruning(t, mus_before, sigmas_before, mus, Sigmas, weights_before, normalized_weights)
    
    
  return mmllhs

def model_marginal_llh_analytic_2x1D_PF_randomized(data, sigma_r, Sigma_0 = 1., N = 64, N2 = 64, verbose = True):
  
  global frequencies
  global kell_pruning
  global chosen_weights
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  normalized_weights = [1.]
  mus = [[0., 0.]]
  Sigmas = [[Sigma_0, Sigma_0]]
  mmllh = 1.
  
  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listaban

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)

  for t in range(1, T + 1): # t darabszam
    
    # muk, sigmak, weightek a prune-olas elott:
    # normalizalatlanul megy ki before
    weights_before = deepcopy(normalized_weights)
    mus_before = deepcopy(mus)
    sigmas_before = deepcopy(Sigmas)

    if len(mus) > N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 10
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind = N2
      while ind != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind]))
        ind += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind])
      value_counts = samples_from_categ_dist.value_counts()
      chosen_weights = dict()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      for indd in range(len(indices)):
        key = str(normalized_weights[indices[indd]])
        if key in list(chosen_weights.keys()):
          chosen_weights[key + '_'] = frequencies[indd]
        else:
          chosen_weights[key] = frequencies[indd]

      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])

  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    

    zt = z[t-1]
    rt = r[t-1]
    
    for k in range(len(mus)):
      
      #sulymodositas
      if k <= int(len(mus)/2) - 1: # xhez soroljuk az ujat
        Sigma_star_inv = zt[1]**2/sigma_r**2
        sigma_inv = 1/Sigmas[k][0]
        Sigma_1 = 1/(Sigma_star_inv + sigma_inv)
        mu_1 = Sigma_1 * (zt[1]*rt/sigma_r**2 + sigma_inv * mus[k][0])

        c_1 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf(0., mean = mus[k][0], cov = Sigmas[k][0]) / \
        multivariate_normal.pdf(mu_1, mean = 0., cov = Sigma_1)

        normalized_weights[k] *= c_1
        mus[k][0] = mu_1
        Sigmas[k][0] = Sigma_1
       
        
      else: # yhoz soroljuk az ujat
        Sigma_star_inv = zt[0]**2/sigma_r**2
        sigma_inv = 1/Sigmas[k][1]
        Sigma_2 = 1/(Sigma_star_inv + sigma_inv)
        mu_2 = Sigma_2 * (zt[0]*rt/sigma_r**2 + sigma_inv * mus[k][1])

        c_2 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf(0., mean = mus[k][1], cov = Sigmas[k][1]) / \
        multivariate_normal.pdf(mu_2, mean = 0., cov = Sigma_2)

        normalized_weights[k] *= c_2
        mus[k][1] = mu_2
        Sigmas[k][1] = Sigma_2
    
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)

    
    if verbose:
      pbar.add(1)
   
    # t adatpont eseten posterior plotolasa pruning elott/utan:
    #plot_posteriors_before_after_pruning(t, mus_before, sigmas_before, mus, Sigmas, weights_before, normalized_weights)
    
    
  return mmllh

def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc



def extract_some_data_points(all_data, start, how_many):
  z = all_data['z'][start:start + how_many]
  r = np.array(all_data['r'][start:start + how_many])
  c = all_data['c'][start:start + how_many]
  data = {'z': z, 'r': r, 'c': c}
  return data

def dream_gammas_from_xdata(data, sigma_r, Sigma_0 = 1.):
  zx = data['z']
  rx = np.array(data['r'])
  Tx = zx.shape[0]
  mu, sigma = gamma_posterior_analytic(data, sigma_r, Sigma_0 = Sigma_0, model = 'x')
  post = tfd.Normal(loc = mu, scale = sigma)
  gamma = np.array(post.sample(Tx))
  gammax = np.zeros((gamma.shape[0], 2))
  gammax[:, 1] = gamma
  return gammax

def dream_gammas_from_2D_data(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]])):
  z = data['z']
  r = np.array(data['r'])
  T = z.shape[0]
  mu, sigma = gamma_posterior_analytic(data, sigma_r, Sigma_0 = Sigma_0, model = '2d')
  post = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix=sigma)
  gamma = np.array(post.sample(T))
  return gamma


def generate_data_from_gammas(gammas, T, contexts, Sigma_0 = 1., infness = 'non-inf', z_random = True, z = None):
  if infness == 'non-inf':
    if z_random:
      z = np.array(tfd.MultivariateNormalDiag(loc = [0, 0], scale_diag = [Sigma_0, Sigma_0]).sample(T))
  elif infness == 'inf':
    if z_random:
      z = np.array(tfd.MultivariateNormalDiag(loc = [0, 0], scale_diag = [Sigma_0, Sigma_0]).sample(100*T))
      z = z[(((z[:, 0] < 0) & (z[:, 1] > 0)) | ((z[:, 0] > 0) & (z[:, 1] < 0)))]
  r_noise = tfd.Normal(0, .001).sample(T)
  r_mean = tf.cast(tf.reduce_sum(tf.multiply(gammas,z),1), 'float32')
  r = r_mean + r_noise
  data = {'z': z, 'r': r, 'c': contexts}
  return data

def generate_informative_y_data(Ty, informativeness):
  alpha = 0
  gamma = gamma_from_alpha(alpha)
  z_prior = tfd.MultivariateNormalDiag(loc=[-informativeness,informativeness], scale_diag=[.3,.3]);
  z = np.array(z_prior.sample(Ty))
  r_noise = tfd.Normal(0, .001).sample(Ty)
  r_mean = tf.reduce_sum(tf.multiply(gamma,z),1)
  r = r_mean + r_noise
  datay = {'z':z,'r':r}
 
  return datay


def homogenize_EM(EM):
  gen = data_generator(EM)
  for i, point in enumerate(gen):
    if i == 0:
      first_c = point['c'][0]
    else:
      EM['c'][i] = first_c


def dream_data_from_posterior(model, posterior, how_many = None):
  '''
  structure of posterior:
      model x, y, 1x2D: [mu, Sigma]
      model 2x1D, 2x2D: [mus, Sigmas, normalized_weights, data_point_counter_list]
      model 2x1D_bg, 2x2D_bg: [mus, Sigmas, data_point_counter_list]

  how_many:
      model x, y, 1x2D: integer scalar
      model 2x1D_bg, 2x2D_bg: list of integer scalars
      model 2x1D, 2x2D: None

  (in case of model 2x1D and 2x2D parameter how_many is not needed bc of the dreaming method)
  '''

  if model == 'x' or model == 'y' or model == '1x2D':
    mu, Sigma = posterior
  elif model == "2x1D" or model == "2x2D":
    mus, Sigmas, normalized_weights, data_point_counter_list = posterior
  elif model == "2x1D_bg" or model == "2x2D_bg":
    mus, Sigmas, _ = posterior

  if model == 'x':
    post = tfd.Normal(loc = mu, scale = Sigma)
    gamma = np.array(post.sample(how_many))
    gamma_out = np.zeros((gamma.shape[0], 2))
    gamma_out[:, 1] = gamma
    data_dream = generate_data_from_gammas(gamma_out, how_many, ['90'] * len(gamma_out))
    return data_dream
  elif model == 'y':
    post = tfd.Normal(loc = mu, scale = Sigma)
    gamma = np.array(post.sample(how_many))
    gamma_out = np.zeros((gamma.shape[0], 2))
    gamma_out[:, 0] = gamma
    data_dream = generate_data_from_gammas(gamma_out, how_many, ['0'] * len(gamma_out))
    return data_dream
  elif model == '1x2D':
    global FIRST_CONTEXT
    post = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = Sigma)
    gamma_out = np.array(post.sample(how_many))
    # angles = infer_angles_from_gammas(gamma_out)
    data_dream = generate_data_from_gammas(gamma_out, how_many, how_many * [FIRST_CONTEXT])
    return data_dream
  elif model == '2x1D':
    bernoulli = tfd.Categorical(probs = normalized_weights)
    chosen_particle_idx = bernoulli.sample(1)
    Tx = data_point_counter_list[int(chosen_particle_idx)][0]
    Ty = data_point_counter_list[int(chosen_particle_idx)][1]

    # model x separately
    components_x = []
    for i in range(len(normalized_weights)):
      components_x.append(tfd.Normal(loc = np.float64(mus[i][0]), scale=np.float64(Sigmas[i][0])))
    post_x = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_x)
    gamma_ = np.array(post_x.sample(Tx))
    gammax = np.zeros((gamma_.shape[0], 2))
    gammax[:, 1] = gamma_

    # model y separately
    components_y = []
    for i in range(len(normalized_weights)):
      components_y.append(tfd.Normal(loc = np.float64(mus[i][1]), scale=np.float64(Sigmas[i][1])))
    post_y = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_y)
    gamma_ = np.array(post_y.sample(Ty))
    gammay = np.zeros((gamma_.shape[0], 2))
    gammay[:, 0] = gamma_

    data_dream_x = generate_data_from_gammas(gammax, Tx, ['90'] * len(gammax))
    data_dream_y = generate_data_from_gammas(gammay, Ty, ['0'] * len(gammay))
    data_dream = concatenate_data(data_dream_x, data_dream_y)
    return data_dream
  elif model == '2x2D':
    bernoulli = tfd.Categorical(probs = normalized_weights)
    chosen_particle_idx = bernoulli.sample(1)
    Tx = data_point_counter_list[int(chosen_particle_idx)][0]
    Ty = data_point_counter_list[int(chosen_particle_idx)][1]

    #kulon x-re
    components_x = []
    for i in range(len(normalized_weights)):
      components_x.append(tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[i][0]), covariance_matrix=np.float64(Sigmas[i][0])))
    post_x = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_x)
    
    gammax = np.array(post_x.sample(Tx))
    # angles_x = infer_angles_from_gammas(gammax)

    #kulon y-ra
    components_y = []
    for i in range(len(normalized_weights)):
      components_y.append(tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[i][1]), covariance_matrix=np.float64(Sigmas[i][1])))
    post_y = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_y)
      
    gammay = np.array(post_y.sample(Ty))
    # angles_y = infer_angles_from_gammas(gammay)
    data_dream_x = generate_data_from_gammas(gammax, Tx, angles_x)
    data_dream_y = generate_data_from_gammas(gammay, Ty, angles_y)
    data_dream = concatenate_data(data_dream_x, data_dream_y)
    return data_dream
  elif model == '2x1D_bg':
    Tx = how_many[0]
    Ty = how_many[1]

    # dreaming from x
    post_x = tfd.Normal(loc = np.float64(mus[0]), scale=np.float64(Sigmas[0]))
      
    gamma_ = np.array(post_x.sample(Tx))
    gammax = np.zeros((gamma_.shape[0], 2))
    gammax[:, 1] = gamma_

    # dreaming from y
    post_y = tfd.Normal(loc = np.float64(mus[1]), scale=np.float64(Sigmas[1]))
     
    gamma_ = np.array(post_y.sample(Ty))
    gammay = np.zeros((gamma_.shape[0], 2))
    gammay[:, 0] = gamma_

    data_dream_x = generate_data_from_gammas(gammax, Tx, ['90'] * len(gammax))
    data_dream_y = generate_data_from_gammas(gammay, Ty, ['0'] * len(gammay))
    data_dream = concatenate_data(data_dream_x, data_dream_y)
    return data_dream
  elif model == '2x2D_bg':
    Tx = how_many[0]
    Ty = how_many[1]
    
    # dreaming from task1
    post_x = tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[0]), covariance_matrix=np.float64(Sigmas[0]))
    gammax = np.array(post_x.sample(Tx))
    angles_x = infer_angles_from_gammas(gammax)
    # dreaming from task2
    post_y = tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[1]), covariance_matrix=np.float64(Sigmas[1]))
    gammay = np.array(post_y.sample(Ty))
    angles_y = infer_angles_from_gammas(gammay)
    data_dream_x = generate_data_from_gammas(gammax, Tx, angles_x)
    data_dream_y = generate_data_from_gammas(gammay, Ty, angles_y)
    data_dream = concatenate_data(data_dream_x, data_dream_y)
    return data_dream

def GR(learning_dict, how_many = None):
  '''
  structure of posterior:
      model x, y, 1x2D: [mu, Sigma]
      model 2x1D, 2x2D: [mus, Sigmas, normalized_weights, data_point_counter_list]
      model 2x1D_bg, 2x2D_bg: [mus, Sigmas, data_point_counter_list]

  how_many:
      model x, y, 1x2D: integer scalar
      model 2x1D_bg, 2x2D_bg: list of integer scalars
      model 2x1D, 2x2D: None

  (in case of model 2x1D and 2x2D parameter how_many is not needed bc of the dreaming method)
  '''
  model = learning_dict['prominent_models'][-1]
  posterior = learning_dict[model]['posteriors'][-1]
  if model == 'x' or model == 'y' or model == '1x2D':
    mu, Sigma = posterior
  elif model == "2x1D" or model == "2x2D":
    mus, Sigmas, normalized_weights, data_point_counter_list = posterior
  elif model == "2x1D_bg" or model == "2x2D_bg":
    mus, Sigmas, _ = posterior

  if model == 'x':
    post = tfd.Normal(loc = mu, scale = Sigma)
    gamma = np.array(post.sample(how_many))
    gamma_out = np.zeros((gamma.shape[0], 2))
    gamma_out[:, 1] = gamma
    data_dream = generate_data_from_gammas(gamma_out, how_many, ['90'] * len(gamma_out))
    return data_dream
  elif model == 'y':
    post = tfd.Normal(loc = mu, scale = Sigma)
    gamma = np.array(post.sample(how_many))
    gamma_out = np.zeros((gamma.shape[0], 2))
    gamma_out[:, 0] = gamma
    data_dream = generate_data_from_gammas(gamma_out, how_many, ['0'] * len(gamma_out))
    return data_dream
  elif model == '1x2D':
    global FIRST_CONTEXT
    post = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = Sigma)
    gamma_out = np.array(post.sample(how_many))
    # angles = infer_angles_from_gammas(gamma_out)
    data_dream = generate_data_from_gammas(gamma_out, how_many, how_many * [FIRST_CONTEXT])
    return data_dream
  elif model == '2x1D':
    bernoulli = tfd.Categorical(probs = normalized_weights)
    chosen_particle_idx = bernoulli.sample(1)
    Tx = data_point_counter_list[int(chosen_particle_idx)][0]
    Ty = data_point_counter_list[int(chosen_particle_idx)][1]

    # model x separately
    components_x = []
    for i in range(len(normalized_weights)):
      components_x.append(tfd.Normal(loc = np.float64(mus[i][0]), scale=np.float64(Sigmas[i][0])))
    post_x = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_x)
    gamma_ = np.array(post_x.sample(Tx))
    gammax = np.zeros((gamma_.shape[0], 2))
    gammax[:, 1] = gamma_

    # model y separately
    components_y = []
    for i in range(len(normalized_weights)):
      components_y.append(tfd.Normal(loc = np.float64(mus[i][1]), scale=np.float64(Sigmas[i][1])))
    post_y = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_y)
    gamma_ = np.array(post_y.sample(Ty))
    gammay = np.zeros((gamma_.shape[0], 2))
    gammay[:, 0] = gamma_

    data_dream_x = generate_data_from_gammas(gammax, Tx, ['90'] * len(gammax))
    data_dream_y = generate_data_from_gammas(gammay, Ty, ['0'] * len(gammay))
    data_dream = concatenate_data([data_dream_x, data_dream_y])
    return data_dream
  elif model == '2x2D':
    bernoulli = tfd.Categorical(probs = normalized_weights)
    chosen_particle_idx = bernoulli.sample(1)
    Tx = data_point_counter_list[int(chosen_particle_idx)][0]
    Ty = data_point_counter_list[int(chosen_particle_idx)][1]

    #kulon x-re
    components_x = []
    for i in range(len(normalized_weights)):
      components_x.append(tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[i][0]), covariance_matrix=np.float64(Sigmas[i][0])))
    post_x = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_x)
    
    gammax = np.array(post_x.sample(Tx))
    angles_x = infer_angles_from_gammas(gammax)

    #kulon y-ra
    components_y = []
    for i in range(len(normalized_weights)):
      components_y.append(tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[i][1]), covariance_matrix=np.float64(Sigmas[i][1])))
    post_y = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_y)
      
    gammay = np.array(post_y.sample(Ty))
    angles_y = infer_angles_from_gammas(gammay)
    data_dream_x = generate_data_from_gammas(gammax, Tx, angles_x)
    data_dream_y = generate_data_from_gammas(gammay, Ty, angles_y)
    data_dream = concatenate_data([data_dream_x, data_dream_y])
    return data_dream
  elif model == '2x1D_bg':
    Tx = how_many[0]
    Ty = how_many[1]

    # dreaming from x
    post_x = tfd.Normal(loc = np.float64(mus[0]), scale=np.float64(Sigmas[0]))
      
    gamma_ = np.array(post_x.sample(Tx))
    gammax = np.zeros((gamma_.shape[0], 2))
    gammax[:, 1] = gamma_

    # dreaming from y
    post_y = tfd.Normal(loc = np.float64(mus[1]), scale=np.float64(Sigmas[1]))
     
    gamma_ = np.array(post_y.sample(Ty))
    gammay = np.zeros((gamma_.shape[0], 2))
    gammay[:, 0] = gamma_

    data_dream_x = generate_data_from_gammas(gammax, Tx, ['90'] * len(gammax))
    data_dream_y = generate_data_from_gammas(gammay, Ty, ['0'] * len(gammay))
    data_dream = concatenate_data([data_dream_x, data_dream_y])
    return data_dream
  elif model == '2x2D_bg':
    Tx = how_many[0]
    Ty = how_many[1]
    
    # dreaming from task1
    post_x = tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[0]), covariance_matrix=np.float64(Sigmas[0]))
    gammax = np.array(post_x.sample(Tx))
    #angles_x = infer_angles_from_gammas(gammax)
    # dreaming from task2
    post_y = tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[1]), covariance_matrix=np.float64(Sigmas[1]))
    gammay = np.array(post_y.sample(Ty))
    #angles_y = infer_angles_from_gammas(gammay)
    if Tx:
        data_dream_x = generate_data_from_gammas(gammax, Tx, [str(task_angles_in_data[0])] * Tx)
    if Ty:
        data_dream_y = generate_data_from_gammas(gammay, Ty, [str(task_angles_in_data[1])] * Ty)
    data_dream = concatenate_data([data_dream_x, data_dream_y])

    return data_dream

