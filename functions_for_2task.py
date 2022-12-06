import helper_mate
import numpy as np
import tensorflow as tf
from copy import deepcopy
from scipy.stats import multivariate_normal
import pandas as pd
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp



def calc_mmllh_1task(data, sigma_r, model, Sigma_0 = None, evaluation = "full", posterior = None, mmllh_prev = None, ret_all_mmllhs = False):
  '''
  calculates the mmLLH, the posterior and the predicitive probability for models involving 1 task
  default values have to be used concurrently, this setting corresponds to full evaluation

  Parameters
  ----------
  data :  dict
        dictionary containing all the information needed for completely define the dataset
        keys : 'z', 'r', 'c'
        values : np.ndarray (latent vectors), tensorflow.Tensor (rewards), list (contexts)
  sigma_r : float
  Sigma_0 : np.ndarray, float
  model : str
        its value can be "x", "y" or "1x2D"
  evaluation : str, optional
        its value can be "full" or "iterative"
        in the latter case posterior and mmllh_prev has to be specified
  Returns
  ----------
  mmllh : float
  mu : numpy.ndarray
  Sigma : numpy.ndarray
  pred prob : float 
      only returned when evaluation = "iterative"
  '''
  if Sigma_0 is None:
    if model == "1x2D":
      Sigma_0 = np.array([[1., 0.],
                          [0., 1.]])
    elif model == "x" or model == "y":
      Sigma_0 = 1.

  if evaluation == "full":
    mmllh_list = []
    zs = data['z']
    rs = np.array(data['r'])
    if zs.size != 0:
      T = np.size(zs,0)
      if model == '1x2D':
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
            mmllh_list.append(y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i))
        posterior = (mu_i, Sigma_i)
        if ret_all_mmllhs:
            return mmllh_list, posterior
        else:
            return y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i), posterior
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
            mmllh_list.append(y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i))
        posterior = (mu_i, Sigma_i)
        if ret_all_mmllhs:
            return mmllh_list, posterior
        else:
            return y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i), posterior
    else:
      if model == '1x2D':
        post = ([0, 0], Sigma_0)
        return 1., post
      else:
        post = (0, Sigma_0)
        return 1., post
  elif evaluation == "iterative":
    if posterior is not None:
      mu_i_prev,\
      Sigma_i_prev = posterior
    else:
      raise Exception("posterior has to be specified if evaluation is iterative")
    if mmllh_prev is not None:
      if model == "1x2D":
        y = mmllh_prev * multivariate_normal.pdf(mu_i_prev, mean = np.array([0,0]), cov = Sigma_i_prev)
      else:
        y = mmllh_prev * multivariate_normal.pdf(mu_i_prev, mean = 0.0, cov = Sigma_i_prev)
    else:
      raise Exception("mmllh_prev has to be specified if evaluation is iterative")
    zs = data['z']
    rs = np.array(data['r'])
    if zs.size != 0:
      T = np.size(zs,0)
      if model == '1x2D':
        assert not np.isscalar(Sigma_0), 'Sigma_0 must be a 2-dimensional array'
        detSigma_0 = np.linalg.det(Sigma_0)
        Sigma_i_inv_prev = np.linalg.inv(Sigma_i_prev)
        
        for t in range(T):
            z = zs[t]
            r = rs[t]
            Sigma_i_star_inv = np.array([[z[0]**2/sigma_r**2, z[0]*z[1]/sigma_r**2],[z[0]*z[1]/sigma_r**2, z[1]**2/sigma_r**2]])
            
            Sigma_i_inv = Sigma_i_star_inv + Sigma_i_inv_prev
            
            Sigma_i = np.linalg.inv(Sigma_i_inv)
            mu_i = Sigma_i.dot(z*r/sigma_r**2 + Sigma_i_inv_prev.dot(mu_i_prev))
            
            y = y * multivariate_normal.pdf(r, mean = 0, cov = sigma_r**2)
            Sigma_i_inv_prev = Sigma_i_inv
            mu_i_prev = mu_i
        
        mmllh = y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i)
        pred_prob = mmllh / mmllh_prev
        post = (mu_i, Sigma_i)
        return mmllh, post, pred_prob
      else:
        '''
        Sigma_0 is the standard deviation of the gamma prior
        '''
        assert np.isscalar(Sigma_0), 'Sigma_0 must be scalar'
        if model == 'x':
          integral_dim = 1
        else:
          integral_dim = 0

        Sigma_i_inv_prev = 1 / Sigma_i_prev
        for t in range(T): #1 iter.
            z = zs[t]
            r = rs[t]
            Sigma_i_star_inv = z[integral_dim]**2/sigma_r**2
            Sigma_i_inv = Sigma_i_star_inv + Sigma_i_inv_prev
            Sigma_i = 1/Sigma_i_inv
            mu_i = Sigma_i * (z[integral_dim]*r/sigma_r**2 + Sigma_i_inv_prev*mu_i_prev)
            y = y * multivariate_normal.pdf(r, mean = 0, cov = sigma_r**2)
            Sigma_i_inv_prev = Sigma_i_inv
            mu_i_prev = mu_i

        mmllh = y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i)
        pred_prob = mmllh / mmllh_prev
        post = (mu_i, Sigma_i)
        return mmllh, post, pred_prob
    else:
      pred_prob = 1.
      post = (mu_i_prev, Sigma_i_prev)
      return mmllh_prev, post, pred_prob


def model_marginal_llh_analytic_2x1D_PF_randomized_posterior_aswell(data, sigma_r, Sigma_0 = 1., N = 64, N2 = 64, verbose = True, ret_all_mmllhs = False):
  
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  normalized_weights = [1.]
  mus = [[0., 0.]]
  Sigmas = [[Sigma_0, Sigma_0]]
  data_point_counter_list = [[0, 0]]


  mmllh = 1.
  
  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listaban

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  mmllh_list = []
  for t in range(1, T + 1): # t darabszam

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 40
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind_prev = N2
      ind_next = N2
      while ind_next != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
        ind_prev = ind_next 
        ind_next += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
      value_counts = samples_from_categ_dist.value_counts()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])
      data_point_counter_list = list(np.array(data_point_counter_list)[indices])
  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    data_point_counter_list = deepcopy(data_point_counter_list) + deepcopy(data_point_counter_list)

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
        data_point_counter_list[k][0] += 1
        
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
        data_point_counter_list[k][1] += 1
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)
    mmllh_list.append(mmllh)
    
    if verbose:
      pbar.add(1)

  # miutan beszortam minden adatot, meg egy subsampling mehet
  if len(mus) > N:
    probs, _ = normalize_weights(normalized_weights)
    factor = 40
    num_samples = int(N2*factor)
    samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
    num_survivors = 0
    ind_prev = N2
    ind_next = N2
    while ind_next != num_samples and num_survivors is not N2:
      num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
      ind_prev = ind_next 
      ind_next += 1

    samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
    value_counts = samples_from_categ_dist.value_counts()
    indices = value_counts.index
    # norm szamitasa
    frequencies = value_counts.values
    normalized_weights, _ = normalize_weights(list(frequencies))
    # megvan a subsampled post.
    mus = list(np.array(mus)[indices])
    Sigmas = list(np.array(Sigmas)[indices])
    data_point_counter_list = list(np.array(data_point_counter_list)[indices])
  if ret_all_mmllhs:
    return mmllh_list, mus, Sigmas, normalized_weights, data_point_counter_list
  else:
    return mmllh, mus, Sigmas, normalized_weights, data_point_counter_list

def model_marginal_llh_analytic_2x2D_PF_randomized_posterior_aswell(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), N = 64, N2 = 64, verbose = True, ret_all_mmllhs = False):
  
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  normalized_weights = [1.]
  mus = [[np.array([0., 0.]), np.array([0., 0.])]]
  Sigmas = [[Sigma_0, Sigma_0]]
  data_point_counter_list = [[0, 0]]


  mmllh = 1.
  
  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listaban

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  mmllh_list = []
  for t in range(1, T + 1): # t darabszam

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 40
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind_prev = N2
      ind_next = N2
      while ind_next != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
        ind_prev = ind_next 
        ind_next += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
      value_counts = samples_from_categ_dist.value_counts()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])
      data_point_counter_list = list(np.array(data_point_counter_list)[indices])
  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    data_point_counter_list = deepcopy(data_point_counter_list) + deepcopy(data_point_counter_list)

    zt = z[t-1]
    rt = r[t-1]
    Sigma_star_inv = np.array([[zt[0]**2/sigma_r**2, zt[0]*zt[1]/sigma_r**2],[zt[0]*zt[1]/sigma_r**2, zt[1]**2/sigma_r**2]])
    for k in range(len(mus)):
      
      #sulymodositas
      if k <= int(len(mus)/2) - 1: # xhez soroljuk az ujat
        
        sigma_inv = np.linalg.inv(Sigmas[k][0])
        Sigma_1 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_1 = np.matmul(Sigma_1, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][0]))

        c_1 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][0], cov = Sigmas[k][0]) / \
        multivariate_normal.pdf(mu_1, mean = [0., 0.], cov = Sigma_1)
        normalized_weights[k] *= c_1
        mus[k][0] = mu_1
        Sigmas[k][0] = Sigma_1
        data_point_counter_list[k][0] += 1
        
      else: # yhoz soroljuk az ujat
        
        sigma_inv = np.linalg.inv(Sigmas[k][1])
        Sigma_2 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_2 = np.matmul(Sigma_2, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][1]))

        c_2 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][1], cov = Sigmas[k][1]) / \
        multivariate_normal.pdf(mu_2, mean = [0., 0.], cov = Sigma_2)

        normalized_weights[k] *= c_2
        mus[k][1] = mu_2
        Sigmas[k][1] = Sigma_2
        data_point_counter_list[k][1] += 1
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)
    mmllh_list.append(mmllh)
    
    if verbose:
      pbar.add(1)

  # miutan beszortam minden adatot, meg egy subsampling mehet
  if len(mus) > N:
    probs, _ = normalize_weights(normalized_weights)
    factor = 40
    num_samples = int(N2*factor)
    samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
    num_survivors = 0
    ind_prev = N2
    ind_next = N2
    while ind_next != num_samples and num_survivors is not N2:
      num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
      ind_prev = ind_next 
      ind_next += 1

    samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
    value_counts = samples_from_categ_dist.value_counts()
    indices = value_counts.index
    # norm szamitasa
    frequencies = value_counts.values
    normalized_weights, _ = normalize_weights(list(frequencies))
    # megvan a subsampled post.
    mus = list(np.array(mus)[indices])
    Sigmas = list(np.array(Sigmas)[indices])
    data_point_counter_list = list(np.array(data_point_counter_list)[indices])

  if ret_all_mmllhs:
    return mmllh_list, mus, Sigmas, normalized_weights, data_point_counter_list
  else:
    return mmllh, mus, Sigmas, normalized_weights, data_point_counter_list

def normalize_weights(weights):
  norm = sum(weights)
  weights = np.array(weights)/norm
  return list(weights), norm

def model_marginal_llh_analytic_posterior_aswell(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), model = '1x2D'):
    # this is the model marginal likelihood function
    # it is validated through 'trial_nonorm_posterior_set_transformed'
    # from that function the only step fowrad is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    zs = data['z']
    rs = np.array(data['r'])
    if zs.size != 0:
      T = np.size(zs,0)
      if model == '1x2D':
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
        return y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i), mu_i, Sigma_i
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

        return y / multivariate_normal.pdf(mu_i, mean = 0.0, cov = Sigma_i), mu_i, Sigma_i
    else:
      return 1.

def mmllh_2x2D_with_background(data, sigma_r, Sigma_0):
  z = data['z']
  r = np.array(data['r'])
  c = data['c']
  distinct_c = np.unique(c)
  assert len(distinct_c) <= 2, "function mmllh_2x2D_with_background(): number of different contexts should be at most 2"
  x_indices = np.where(np.array(c) == distinct_c[0])[0]
  T = z.shape[0]
  
  indices = np.arange(T)
  T1 = len(x_indices)
  T2 = len(indices) - T1
  complementer_subset = [item for item in indices if item not in x_indices]

  if len(x_indices) > 0:
    zx = z[x_indices]
    rx = r[x_indices]
    cx = np.array(c)[x_indices]
    cx = cx[0]
    datax = {'z': zx, 'r': rx}
    mmllh_x, mu_x, sigma_x = model_marginal_llh_analytic_posterior_aswell(datax, sigma_r, Sigma_0 = Sigma_0, model = '1x2D')
  else:
    mu_x = [0., 0.]
    sigma_x = [[1., 0.], [0., 1.]]
    mmllh_x = 1.
    cx = ''

  if len(complementer_subset) > 0:
    zy = z[complementer_subset]
    ry = r[complementer_subset]
    cy = np.array(c)[complementer_subset]
    cy = cy[0]
    datay = {'z': zy, 'r': ry}
    mmllh_y, mu_y, sigma_y = model_marginal_llh_analytic_posterior_aswell(datay, sigma_r, Sigma_0 = Sigma_0, model = '1x2D')
  else:
    mu_y = [0., 0.]
    sigma_y = [[1., 0.], [0., 1.]]
    mmllh_y = 1.
    cy = ''
  conts = [cx, cy]
  if '' in conts:
    conts.remove('')
  return [mmllh_x, mmllh_y], [mu_x, mu_y], [sigma_x, sigma_y], [T1, T2], conts

def mmllh_2x1D_with_background(data, sigma_r, Sigma_0):
  z = data['z']
  r = np.array(data['r'])
  c = data['c']
  x_indices = np.where(np.array(c) == '90')[0]
  T = z.shape[0]
  
  indices = np.arange(T)
  complementer_subset = [item for item in indices if item not in x_indices]

  if len(x_indices) > 0:
    zx = z[x_indices]
    rx = r[x_indices]
    Tx = zx.shape[0]
    cx = '90'
    datax = {'z': zx, 'r': rx}
    mmllh_x, mu_x, sigma_x = model_marginal_llh_analytic_posterior_aswell(datax, sigma_r, Sigma_0 = Sigma_0, model = 'x')
  else:
    mmllh_x = 1.
    mu_x = 0.
    sigma_x = 1.
    cx = ''
    Tx = 0
  if len(complementer_subset) > 0:
    cy = '0'
    zy = z[complementer_subset]
    ry = r[complementer_subset]
    datay = {'z': zy, 'r': ry}
    Ty = zy.shape[0]
    mmllh_y, mu_y, sigma_y = model_marginal_llh_analytic_posterior_aswell(datay, sigma_r, Sigma_0 = Sigma_0, model = 'y')
  else:
    cy = ''
    Ty = 0
    mmllh_y = 1.
    mu_y = 0.
    sigma_y = 1.
  conts = [cx, cy]
  if '' in conts:
    conts.remove('')
  return [mmllh_x, mmllh_y], [mu_x, mu_y], [sigma_x, sigma_y], [Tx, Ty], conts



def mmllh_2x2D_PF_from_posterior(posterior, mmllh_prev, data, sigma_r, Sigma_0, N, N2, verbose):
  
  z = data['z']
  r = np.array(data['r'])
  T = z.shape[0]

  mmllh = mmllh_prev
  mus,\
  Sigmas,\
  normalized_weights,\
  data_point_counter_list = posterior

  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listaban

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)

  for t in range(1, T + 1): # t darabszam

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 40
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind_prev = N2
      ind_next = N2
      while ind_next != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
        ind_prev = ind_next 
        ind_next += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
      value_counts = samples_from_categ_dist.value_counts()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])
      data_point_counter_list = list(np.array(data_point_counter_list)[indices])
  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    data_point_counter_list = deepcopy(data_point_counter_list) + deepcopy(data_point_counter_list)

    zt = z[t-1]
    rt = r[t-1]
    Sigma_star_inv = np.array([[zt[0]**2/sigma_r**2, zt[0]*zt[1]/sigma_r**2],[zt[0]*zt[1]/sigma_r**2, zt[1]**2/sigma_r**2]])
    for k in range(len(mus)):
      
      #sulymodositas
      if k <= int(len(mus)/2) - 1: # xhez soroljuk az ujat
        
        sigma_inv = np.linalg.inv(Sigmas[k][0])
        Sigma_1 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_1 = np.matmul(Sigma_1, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][0]))

        c_1 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][0], cov = Sigmas[k][0]) / \
        multivariate_normal.pdf(mu_1, mean = [0., 0.], cov = Sigma_1)
        normalized_weights[k] *= c_1
        mus[k][0] = mu_1
        Sigmas[k][0] = Sigma_1
        data_point_counter_list[k][0] += 1
        
      else: # yhoz soroljuk az ujat
        
        sigma_inv = np.linalg.inv(Sigmas[k][1])
        Sigma_2 = np.linalg.inv(Sigma_star_inv + sigma_inv)
        mu_2 = np.matmul(Sigma_2, zt * rt/sigma_r**2 + np.matmul(sigma_inv, mus[k][1]))

        c_2 = multivariate_normal.pdf(rt, mean = 0, cov = sigma_r**2) * multivariate_normal.pdf([0., 0.], mean = mus[k][1], cov = Sigmas[k][1]) / \
        multivariate_normal.pdf(mu_2, mean = [0., 0.], cov = Sigma_2)

        normalized_weights[k] *= c_2
        mus[k][1] = mu_2
        Sigmas[k][1] = Sigma_2
        data_point_counter_list[k][1] += 1
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)

    
    if verbose:
      pbar.add(1)

  # miutan beszortam minden adatot, meg egy subsampling mehet
  if len(mus) > N:
    probs, _ = normalize_weights(normalized_weights)
    factor = 40
    num_samples = int(N2*factor)
    samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
    num_survivors = 0
    ind_prev = N2
    ind_next = N2
    while ind_next != num_samples and num_survivors is not N2:
      num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
      ind_prev = ind_next 
      ind_next += 1

    samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
    value_counts = samples_from_categ_dist.value_counts()
    indices = value_counts.index
    # norm szamitasa
    frequencies = value_counts.values
    normalized_weights, _ = normalize_weights(list(frequencies))
    # megvan a subsampled post.
    mus = list(np.array(mus)[indices])
    Sigmas = list(np.array(Sigmas)[indices])
    data_point_counter_list = list(np.array(data_point_counter_list)[indices])
  pp = mmllh / mmllh_prev
  return mmllh, mus, Sigmas, normalized_weights, data_point_counter_list, pp


def mmllh_2x1D_PF_from_posterior(posterior, mmllh_prev, data, sigma_r, Sigma_0, N, N2, verbose):
  """
  posterior: posterior adatai
  data: ezekkel fejlesztjuk tovabb a posteriort
  """
  z = data['z']
  r = np.array(data['r'])
  T = z.shape[0]

  mmllh = mmllh_prev
  mus,\
  Sigmas,\
  normalized_weights,\
  data_point_counter_list = posterior

  normalized_weights_array = [] # minden adatpont hozzaadasakor ehhez hozza lesz fuzve az aktualis weightset egy-egy listaban

  kell_pruning = False # flag, ami egyszer allitodik
  if verbose:
    pbar = tf.keras.utils.Progbar(T)

  for t in range(1, T + 1): # t darabszam

    if len(mus) >= N:
      kell_pruning = True
      
    if kell_pruning:
      # subsampling
      probs, _ = normalize_weights(normalized_weights)
      
      factor = 40
      num_samples = int(N2*factor)
      samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
      num_survivors = 0
      ind_prev = N2
      ind_next = N2
      while ind_next != num_samples and num_survivors is not N2:
        num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
        ind_prev = ind_next 
        ind_next += 1

      samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
      value_counts = samples_from_categ_dist.value_counts()
      indices = value_counts.index
      # norm szamitasa
      frequencies = value_counts.values
      normalized_weights, _ = normalize_weights(list(frequencies))
      # megvan a subsampled post.

      mus = list(np.array(mus)[indices])
      Sigmas = list(np.array(Sigmas)[indices])
      data_point_counter_list = list(np.array(data_point_counter_list)[indices])
  
    mus = deepcopy(mus) + deepcopy(mus)
    Sigmas = deepcopy(Sigmas) + deepcopy(Sigmas)
    normalized_weights = deepcopy(normalized_weights) + deepcopy(normalized_weights)
    data_point_counter_list = deepcopy(data_point_counter_list) + deepcopy(data_point_counter_list)

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
        data_point_counter_list[k][0] += 1
        
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
        data_point_counter_list[k][1] += 1
    
    if kell_pruning:
      norm = sum(normalized_weights)
      mmllh *= norm/2
    else:
      normalized_weights, norm = normalize_weights(normalized_weights)
      mmllh *= (norm/2)

    
    if verbose:
      pbar.add(1)

  # miutan beszortam minden adatot, meg egy subsampling mehet
  if len(mus) > N:
    probs, _ = normalize_weights(normalized_weights)
    factor = 40
    num_samples = int(N2*factor)
    samples_from_categ_dist = tfd.Categorical(probs = probs).sample(num_samples)
    num_survivors = 0
    ind_prev = N2
    ind_next = N2
    while ind_next != num_samples and num_survivors is not N2:
      num_survivors = len(np.unique(samples_from_categ_dist[:ind_next]))
      ind_prev = ind_next 
      ind_next += 1

    samples_from_categ_dist = pd.Series(samples_from_categ_dist[:ind_prev])
    value_counts = samples_from_categ_dist.value_counts()
    indices = value_counts.index
    # norm szamitasa
    frequencies = value_counts.values
    normalized_weights, _ = normalize_weights(list(frequencies))
    # megvan a subsampled post.
    mus = list(np.array(mus)[indices])
    Sigmas = list(np.array(Sigmas)[indices])
    data_point_counter_list = list(np.array(data_point_counter_list)[indices])
  pp = mmllh / mmllh_prev
  return mmllh, mus, Sigmas, normalized_weights, data_point_counter_list, pp

def filter_based_on_context(data, context):
  c = np.array(data['c'])
  z = data['z']
  r = np.array(data['r'])
  idxs = np.where(c == context)[0]
  if len(idxs) > 0:
    return {'z': z[idxs], 'r': r[idxs], 'c': list(c[idxs])}
  else:
    return None

def exclude_context(data, context):
  c = np.array(data['c'])
  z = data['z']
  r = np.array(data['r'])
  idxs = np.where(c != context)[0]
  if len(idxs) > 0:
    return {'z': z[idxs], 'r': r[idxs], 'c': list(c[idxs])}
  else:
    return None

def retrieve_unique_contexts(data):
  c = data['c']
  return list(np.unique(c))


def mmllh_2x2D_bg_from_posterior(posterior, mmllhs_prev, prev_contexts, data, sigma_r, Sigma_0):
  '''
  len(prev_contexts) == 2  --> prev_contexts should coincide with the distinct contexts in data
  '''
  if len(data['z']) == 0:
    return mmllhs_prev, posterior, prev_contexts
  if len(prev_contexts) == 0:  # there wasn't any context before, full evaluation
    mmllhs, *posterior, new_contexts = \
    mmllh_2x2D_with_background(data, sigma_r, Sigma_0)
    return mmllhs, posterior, new_contexts
  elif len(prev_contexts) == 1:  # T1 or T2 is zero in this case
    mmllh_previous = mmllhs_prev[0] * mmllhs_prev[1]
    if len(mmllhs_prev) == 2:
      assert 1 in mmllhs_prev, "in case of 1 prev context, one of the prev mmllhs should be 1."
    mus_prev,\
    sigmas_prev,\
    Ts_prev = posterior
    sorted_zip_accord_to_T = sorted(zip(mus_prev, sigmas_prev, Ts_prev), key = lambda x: x[-1])
    mus_prev, sigmas_prev, Ts_prev = list(zip(*sorted_zip_accord_to_T))
    # 0 is the first element in Ts_prev from now on
    mmllhs_prev_copy = list(mmllhs_prev).copy()  # maybe one of the elements in mmllhs_prev_copy is 1., but giving a one-element list is also acceptable
    if 1 in mmllhs_prev_copy:
      mmllhs_prev_copy.remove(1)
    mmllh_prev = mmllhs_prev_copy[0]
    prev_context = prev_contexts[0]
    data_with_prev_cont = filter_based_on_context(data, prev_context)
    if data_with_prev_cont is not None:
      T1 = Ts_prev[-1] + len(data_with_prev_cont['z'])
      post_prev = (mus_prev[1], sigmas_prev[1])
      mmllh_1_new, post1, _ = calc_mmllh_1task(data_with_prev_cont, sigma_r, model = "1x2D", Sigma_0 = Sigma_0, evaluation = "iterative", posterior = post_prev, mmllh_prev = mmllh_prev)
      mu_1_new, sigma_1_new = post1
    else:
      mu_1_new, sigma_1_new, T1 = (mus_prev[1], sigmas_prev[1], Ts_prev[1])
      T1 = Ts_prev[-1]
      mmllh_1_new = mmllh_prev

    other_part_of_data = exclude_context(data, prev_context)
    if other_part_of_data is not None:
      T2 = len(other_part_of_data['z'])
      new_context = retrieve_unique_contexts(other_part_of_data)[0]  # function returns a list of unique contexts
      mmllh_2_new, post2 = calc_mmllh_1task(other_part_of_data, sigma_r, model = "1x2D", Sigma_0 = Sigma_0, evaluation = "full")
      mu_2_new, sigma_2_new = post2
    else:
      mu_2_new, sigma_2_new = (mus_prev[0], sigmas_prev[0])
      T2 = 0
      mmllh_2_new = 1.
      new_context = ''
    post_ret = [[mu_1_new, mu_2_new], [sigma_1_new, sigma_2_new], [T1, T2]]
    cont_list = [prev_context, new_context]
    if '' in cont_list:
      cont_list.remove('')
    mmllh_new = mmllh_1_new * mmllh_2_new
    pred_prob = mmllh_new / mmllh_previous
    return [mmllh_1_new, mmllh_2_new], post_ret, cont_list, pred_prob
  
  elif len(prev_contexts) == 2:
    mmllh_previous = mmllhs_prev[0] * mmllhs_prev[1]
    mus_prev,\
    sigmas_prev,\
    Ts_prev = posterior
    mmllhs = []
    mus = []
    sigmas = []
    Ts = []
    for i, prev_context in enumerate(prev_contexts):
      data_filtered = filter_based_on_context(data, prev_context)
      if data_filtered is not None:
        post_prev = (mus_prev[i], sigmas_prev[i])
        mmllh_new, post_, _ = calc_mmllh_1task(data_filtered, sigma_r, model = "1x2D", Sigma_0 = Sigma_0, evaluation = "iterative", posterior = post_prev, mmllh_prev = mmllhs_prev[i])
        mu_new, sigma_new = post_
        mmllhs.append(mmllh_new)
        mus.append(mu_new)
        sigmas.append(sigma_new)
        Ts.append(Ts_prev[i] + len(data_filtered['z']))
      else:
        mmllhs.append(mmllhs_prev[i])
        mus.append(mus_prev[i])
        sigmas.append(sigmas_prev[i])
        Ts.append(Ts_prev[i])
    post_ret = [mus, sigmas, Ts]
    mmllh_new = mmllhs[0] * mmllhs[1]
    pred_prob = mmllh_new / mmllh_previous
    return mmllhs, post_ret, prev_contexts, pred_prob
      
  
  

      
  
  

      
def mmllh_2x1D_bg_from_posterior(posterior, mmllhs_prev, prev_contexts, data, sigma_r, Sigma_0):
  '''
  len(prev_contexts) == 2  --> prev_contexts should coincide with the distinct contexts in data
  '''
  if len(data['z']) == 0:
    return mmllhs_prev, posterior, prev_contexts
  if len(prev_contexts) == 0:  # there wasn't any context before, full evaluation
    mmllhs, *posterior, new_contexts = \
    mmllh_2x1D_with_background(data, sigma_r, Sigma_0)
    return mmllhs, posterior, new_contexts
  elif len(prev_contexts) == 1:  # T1 or T2 is zero in this case
    mmllh_previous = mmllhs_prev[0] * mmllhs_prev[1]
    if len(mmllhs_prev) == 2:
      assert 1 in mmllhs_prev, "in case of 1 prev context, one of the prev mmllhs should be 1."
    mus_prev,\
    sigmas_prev,\
    Ts_prev = posterior
    sorted_zip_accord_to_T = sorted(zip(mus_prev, sigmas_prev, Ts_prev), key = lambda x: x[-1])
    mus_prev, sigmas_prev, Ts_prev = list(zip(*sorted_zip_accord_to_T))
    # 0 is the first element in Ts_prev from now on
    mmllhs_prev_copy = list(mmllhs_prev).copy()  # maybe one of the elements in mmllhs_prev_copy is 1., but giving a one-element list is also acceptable
    if 1 in mmllhs_prev_copy:
      mmllhs_prev_copy.remove(1)
    mmllh_prev = mmllhs_prev_copy[0]
    prev_context = prev_contexts[0]
    x_is_the_prev = False
    if prev_context == '90':
        x_is_the_prev = True
    data_with_prev_cont = filter_based_on_context(data, prev_context)
    if data_with_prev_cont is not None:
      T1 = Ts_prev[-1] + len(data_with_prev_cont['z'])
      post_prev = (mus_prev[1], sigmas_prev[1])
      if prev_context == '0':
        mmllh_1_new, post1, _ = calc_mmllh_1task(data_with_prev_cont, sigma_r, model = "y", Sigma_0 = Sigma_0, evaluation = "iterative", posterior = post_prev, mmllh_prev = mmllh_prev)
      elif prev_context == '90':
        mmllh_1_new, post1, _ = calc_mmllh_1task(data_with_prev_cont, sigma_r, model = "x", Sigma_0 = Sigma_0, evaluation = "iterative", posterior = post_prev, mmllh_prev = mmllh_prev)
      mu_1_new, sigma_1_new = post1
    else:
      mu_1_new, sigma_1_new, T1 = (mus_prev[1], sigmas_prev[1], Ts_prev[1])
      T1 = Ts_prev[-1]
      mmllh_1_new = mmllh_prev

    other_part_of_data = exclude_context(data, prev_context)
    if other_part_of_data is not None:
      T2 = len(other_part_of_data['z'])
      new_context = retrieve_unique_contexts(other_part_of_data)[0]  # function returns a list of unique contexts
      if new_context == '0':
        mmllh_2_new, post2 = calc_mmllh_1task(other_part_of_data, sigma_r, model = "y", Sigma_0 = Sigma_0, evaluation = "full")
      elif new_context == '90':
        mmllh_2_new, post2 = calc_mmllh_1task(other_part_of_data, sigma_r, model = "x", Sigma_0 = Sigma_0, evaluation = "full")
      mu_2_new, sigma_2_new = post2
    else:
      mu_2_new, sigma_2_new = (mus_prev[0], sigmas_prev[0])
      T2 = 0
      mmllh_2_new = 1.
      new_context = ''
    if x_is_the_prev:
        post_ret = [[mu_1_new, mu_2_new], [sigma_1_new, sigma_2_new], [T1, T2]]
        cont_list = [prev_context, new_context]
        mmllhs_returned = [mmllh_1_new, mmllh_2_new]
    else:
        post_ret = [[mu_2_new, mu_1_new], [sigma_2_new, sigma_1_new], [T2, T1]]
        cont_list = [new_context, prev_context]
        mmllhs_returned = [mmllh_2_new, mmllh_1_new]
    if '' in cont_list:
      cont_list.remove('')
    mmllh_new = mmllh_1_new * mmllh_2_new
    pred_prob = mmllh_new / mmllh_previous
    return mmllhs_returned, post_ret, cont_list, pred_prob
  
  elif len(prev_contexts) == 2:
    mmllh_previous = mmllhs_prev[0] * mmllhs_prev[1]
    mus_prev,\
    sigmas_prev,\
    Ts_prev = posterior
    mmllhs = []
    mus = []
    sigmas = []
    Ts = []
    for i, prev_context in enumerate(prev_contexts):
      data_filtered = filter_based_on_context(data, prev_context)
      if data_filtered is not None:
        post_prev = (mus_prev[i], sigmas_prev[i])
        if prev_context == '0':
          mmllh_new, post_, _ = calc_mmllh_1task(data_filtered, sigma_r, model = "y", Sigma_0 = Sigma_0, evaluation = "iterative", posterior = post_prev, mmllh_prev = mmllhs_prev[i])
        elif prev_context == '90':
          mmllh_new, post_, _ = calc_mmllh_1task(data_filtered, sigma_r, model = "x", Sigma_0 = Sigma_0, evaluation = "iterative", posterior = post_prev, mmllh_prev = mmllhs_prev[i])
        mu_new, sigma_new = post_
        mmllhs.append(mmllh_new)
        mus.append(mu_new)
        sigmas.append(sigma_new)
        Ts.append(Ts_prev[i] + len(data_filtered['z']))
      else:
        mmllhs.append(mmllhs_prev[i])
        mus.append(mus_prev[i])
        sigmas.append(sigmas_prev[i])
        Ts.append(Ts_prev[i])

    post_ret = [mus, sigmas, Ts]
    mmllh_new = mmllhs[0] * mmllhs[1]
    pred_prob = mmllh_new / mmllh_previous
    return mmllhs, post_ret, prev_contexts, pred_prob
      
  
  

      
  
  


  
      
  
  



def calc_mmllh_2task(data, sigma_r, model, Sigma_0 = None, evaluation = "full", num_particles = 64, verbose = False, marginalize = True, posterior_prev = None, mmllh_prev = None, mmllhs_prev = None, prev_contexts = None, ret_all_mmllhs = False):
  '''
  calculates the mmLLH, the posterior and the predicitive probability for models involving 2 task
  default values have to be used concurrently, this setting corresponds to full evaluation

  Parameters
  ----------
  data :  dict
        dictionary containing all the information needed for completely define the dataset
        keys : 'z', 'r', 'c'
        values : np.ndarray (latent vectors), tensorflow.Tensor (rewards), list (contexts)
  sigma_r : float
  marginalize : boolean
        determines the need for marginalization over task identities
  num_particles : int
        in case marginalization is needed, this determines the number of particles of which PF make use
  Sigma_0 : np.ndarray, float, optional
  posterior_prev : tuple or list
      mus : list of lists of numpy.ndarrays
          list of expected values of the particles, each and every list item is another list containing mu for task1 and task2
      Sigmas : list of lists of numpy.ndarrays
          list of covariance matrices of the particles, each and every list item is another list containing Sigma for task1 and task2
      normalized_weights : list
          normalized weights of the particles in the MoG posterior
      data_point_counter_list : 
          list of lists, its items correspond to the number of data points incorporated in each particle
  mmllh_prev : float
      mmllh produced during the calculation of posterior_prev 
  model : str
        its value can be "2x1D" or "2x2D"
  evaluation : str, optional
        its value can be "full" or "iterative"
        in the latter case posterior and mmllh_prev has to be specified
  verbose : boolean
        in case of higher verbosity a progress bar appears in some cases (this functionality will be elaborated in the sequel) during the calculation 
  Returns
  ----------
  mmllh : float
  posterior : tuple
      mus : list of lists of numpy.ndarrays
          list of expected values of the particles, each and every list item is another list containing mu for task1 and task2
      Sigmas : list of lists of numpy.ndarrays
          list of covariance matrices of the particles, each and every list item is another list containing Sigma for task1 and task2
      normalized_weights : list
          normalized weights of the particles in the MoG posterior
      data_point_counter_list : 
          list of lists, its items correspond to the number of data points incorporated in each particle
  '''
  if Sigma_0 is None:
    if model == "2x2D":
      Sigma_0 = np.array([[1., 0.],
                          [0., 1.]])
    elif model == "2x1D":
      Sigma_0 = 1.
  if evaluation == "full":
    if model == "2x1D":
      if marginalize:
        if ret_all_mmllhs:
            mmllh_list, mus, Sigmas, normalized_weights, data_point_counter_list = \
            model_marginal_llh_analytic_2x1D_PF_randomized_posterior_aswell(data, 
                                                                            sigma_r, 
                                                                            Sigma_0 = Sigma_0, 
                                                                            N = num_particles, 
                                                                            N2 = num_particles, 
                                                                            verbose = verbose,
                                                                            ret_all_mmllhs = True)
            posterior = (mus, Sigmas, normalized_weights, data_point_counter_list)
            return mmllh_list, posterior
        else:
            mmllh, mus, Sigmas, normalized_weights, data_point_counter_list = \
            model_marginal_llh_analytic_2x1D_PF_randomized_posterior_aswell(data, 
                                                                            sigma_r, 
                                                                            Sigma_0 = Sigma_0, 
                                                                            N = num_particles, 
                                                                            N2 = num_particles, 
                                                                            verbose = verbose)
            posterior = (mus, Sigmas, normalized_weights, data_point_counter_list)
            return mmllh, posterior
      else:
        mmllhs, mus, Sigmas, data_point_counter_list, contexts = \
        mmllh_2x1D_with_background(data, sigma_r, Sigma_0)
        posterior = (mus, Sigmas, data_point_counter_list)
        return mmllhs, posterior, contexts
    elif model == "2x2D":
      if marginalize:
        if ret_all_mmllhs:
            mmllh_list, mus, Sigmas, normalized_weights, data_point_counter_list = \
            model_marginal_llh_analytic_2x2D_PF_randomized_posterior_aswell(data, 
                                                                            sigma_r, 
                                                                            Sigma_0 = Sigma_0, 
                                                                            N = num_particles, 
                                                                            N2 = num_particles, 
                                                                            verbose = verbose,
                                                                            ret_all_mmllhs = True)
            posterior = (mus, Sigmas, normalized_weights, data_point_counter_list)
            return mmllh_list, posterior
        else:
            mmllh, mus, Sigmas, normalized_weights, data_point_counter_list = \
            model_marginal_llh_analytic_2x2D_PF_randomized_posterior_aswell(data, 
                                                                            sigma_r, 
                                                                            Sigma_0 = Sigma_0, 
                                                                            N = num_particles, 
                                                                            N2 = num_particles, 
                                                                            verbose = verbose)
            posterior = (mus, Sigmas, normalized_weights, data_point_counter_list)
            return mmllh, posterior
      else:
        mmllhs, mus, Sigmas, data_point_counter_list, contexts = \
        mmllh_2x2D_with_background(data, sigma_r, Sigma_0)
        posterior = (mus, Sigmas, data_point_counter_list)
        return mmllhs, posterior, contexts
  elif evaluation == "iterative":
    if model == "2x1D":
      if marginalize:
        mmllh, *posterior, pp = \
        mmllh_2x1D_PF_from_posterior(posterior_prev,
                                     mmllh_prev,
                                    data,
                                    sigma_r,
                                    Sigma_0 = Sigma_0,
                                    N = num_particles, N2 = num_particles,
                                    verbose = verbose)
        return mmllh, posterior, pp
      else:
        mmllhs, posterior, contexts, pred_prob = mmllh_2x1D_bg_from_posterior(posterior_prev, mmllhs_prev, prev_contexts, data, sigma_r, Sigma_0)
        return mmllhs, posterior, contexts, pred_prob

    elif model == "2x2D":
      if marginalize:
        mmllh, *posterior, pp = \
        mmllh_2x2D_PF_from_posterior(posterior_prev,
                                     mmllh_prev,
                                    data,
                                    sigma_r,
                                    Sigma_0 = Sigma_0,
                                    N = num_particles, N2 = num_particles,
                                    verbose = verbose)
        return mmllh, posterior, pp
      else:
        mmllhs, posterior, contexts, pp = mmllh_2x2D_bg_from_posterior(posterior_prev, mmllhs_prev, prev_contexts, data, sigma_r, Sigma_0)
        return mmllhs, posterior, contexts, pp





