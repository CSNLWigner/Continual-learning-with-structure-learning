def mmllh_2x1D_with_background(data, sigma_r, x_indices):
  z = data['z']
  r = np.array(data['r'])
  T = z.shape[0]
  
  indices = np.arange(T)
  complementer_subset = [item for item in indices if item not in x_indices]

  if len(x_indices) > 0:
    zx = z[x_indices]
    rx = r[x_indices]
    datax = {'z': zx, 'r': rx}
    mmllh_x, mu_x, sigma_x = model_marginal_llh_analytic_posterior_aswell(datax, sigma_r, Sigma_0 = 1., model = 'x')
  else:
    mmllh_x = 1.
    mu_x = 0.
    sigma_x = 1.

  if len(complementer_subset) > 0:
    zy = z[complementer_subset]
    ry = r[complementer_subset]
    datay = {'z': zy, 'r': ry}
    mmllh_y, mu_y, sigma_y = model_marginal_llh_analytic_posterior_aswell(datay, sigma_r, Sigma_0 = 1., model = 'y')
  else:
    mmllh_y = 1.
    mu_y = 0.
    sigma_y = 1.

  return mmllh_x * mmllh_y, (mu_x, mu_y), (sigma_x, sigma_y)


def model_marginal_llh_analytic_2x1D_PF_randomized_posterior_aswell(data, sigma_r, Sigma_0 = 1., N = 64, N2 = 64, verbose = True):
  
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

  return mmllh, mus, Sigmas, normalized_weights, data_point_counter_list

def mmllh_2x1D_PF_from_posterior(posterior, data, sigma_r, Sigma_0 = 1., N = 64, N2 = 64, verbose = True):
  """
  posterior: posterior adatai
  data: ezekkel fejlesztjuk tovabb a posteriort
  """
  z = data['z']
  r = np.array(data['r'])
  # inicializalas
  T = z.shape[0]
  # ezen a ponton 0 db adatpontot vettem figyelembe, ezek a prior adatai, ezek frissulnek majd iterativan:
  #normalized_weights = [1.]
  #mus = [[0., 0.]]
  #Sigmas = [[Sigma_0, Sigma_0]]
  #data_point_counter_list = [[0, 0]]
  mmllh,\
  mus,\
  Sigmas,\
  normalized_weights,\
  data_point_counter_list = posterior

  #mmllh = 1.
  
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

  return mmllh, mus, Sigmas, normalized_weights, data_point_counter_list
