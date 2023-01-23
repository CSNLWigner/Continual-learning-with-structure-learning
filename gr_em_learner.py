import functions_for_2task as f
import helper_mate as h
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import rcParams
rcParams['font.size'] = 15
rcParams['figure.figsize'] = (15, 10)
Sigma_0_1task_def = 1.
Sigma_0_2task_def = np.array([[1., 0.],
                              [0., 1.]])

def size_of_data(data):
  return len(data['z'])
  
def task_complexity(model):
  if model in ['x', 'y', '1x2D']:
    return '1_task'
  elif model in ['2x1D', '2x2D', '2x1D_bg', '2x2D_bg']:
    return '2_task'


def fill_learning_dict(learning_dict, T, param_name, param_value, model = None, param_is_separate = False):
  if not param_is_separate:
    param_list_len = len(learning_dict[model][param_name])
    if param_list_len < T - 1:
      length_diff = (T - 1) - param_list_len
      learning_dict[model][param_name].extend([None] * length_diff)
      learning_dict[model][param_name].append(param_value)
    elif param_list_len == T - 1:
      learning_dict[model][param_name].append(param_value)
    else:
      raise Exception("length of posteriors list exceeds T - 1")
  else:
    param_list_len = len(learning_dict[param_name])
    if param_list_len < T - 1:
      length_diff = (T - 1) - param_list_len
      learning_dict[param_name].extend([None] * length_diff)
      learning_dict[param_name].append(param_value)
    elif param_list_len == T - 1:
      learning_dict[param_name].append(param_value)
    else:
      raise Exception("length of params list exceeds T - 1")
      
def evaluate_all_full(data, learning_dict, sigma_r, model_set, num_particles, Sigma_0_1task = None, Sigma_0_2task = None):
  '''
  currently intended for use for one data point, full eval
  evaluates all models in learning_dict
  '''
  if Sigma_0_1task is None:
    Sigma_0_1task = Sigma_0_1task_def
  if Sigma_0_2task is None:
    Sigma_0_2task = Sigma_0_2task_def

  T = size_of_data(data)
  context = data['c'][0] # LETS ASSUME, WE HAVE 1 DATA POINT
  for model in model_set:
    complexity = task_complexity(model)
    if complexity == '1_task':
      mmllh, _ = f.calc_mmllh_1task(data, sigma_r, model, evaluation = "full")
      learning_dict[model]['mmllh'][:, T - 1] = mmllh
      if model == '1x2D':
          posterior = ([0, 0], Sigma_0_2task_def)
      else:
          posterior = (0, Sigma_0_1task_def)
      fill_learning_dict(learning_dict, T, 'posteriors', posterior, model)
    elif complexity == '2_task':
      if 'bg' in model:
        mmllhs, _, contexts = f.calc_mmllh_2task(data, sigma_r, model.replace('_bg', ''), marginalize = False, evaluation = "full")
        mmllh = 1.
        for context in mmllhs:
          mmllh *= mmllhs[context]
        fill_learning_dict(learning_dict, T, 'contexts', contexts, model)
        fill_learning_dict(learning_dict, T, 'mmllhs', mmllhs, model)
      else:
        mmllh, _ = f.calc_mmllh_2task(data, sigma_r, model, num_particles = num_particles, evaluation = "full")
      learning_dict[model]['mmllh'][:, T - 1] = mmllh
      if '1D' in model:
        if 'bg' in model:
            mus = {context: 0., 'unknown': 0.}
            Sigmas = {context: Sigma_0_1task_def, 'unknown': Sigma_0_1task_def}
            data_point_counter_list = {context: 0, 'unknown': 0}
            posterior = [mus, Sigmas, data_point_counter_list]
        else:
            normalized_weights = [1.]
            mus = [[0., 0.]]
            Sigmas = [[Sigma_0_1task_def, Sigma_0_1task_def]]
            data_point_counter_list = [[0, 0]]
            posterior = [mus, Sigmas, normalized_weights, data_point_counter_list]
      else:
        if 'bg' in model:
          mus = {context: 0., 'unknown': 0.}
          Sigmas = {context: Sigma_0_2task_def, 'unknown': Sigma_0_2task_def}
          data_point_counter_list = {context: 0, 'unknown': 0}
          posterior = [mus, Sigmas, data_point_counter_list]
        else:
          normalized_weights = [1.]
          mus = [[np.array([0., 0.]), np.array([0., 0.])]]
          Sigmas = [[Sigma_0_2task_def, Sigma_0_2task_def]]
          data_point_counter_list = [[0, 0]]
          posterior = [mus, Sigmas, normalized_weights, data_point_counter_list]
      fill_learning_dict(learning_dict, T, 'posteriors', posterior, model)
  learning_dict, winner_model = who_is_the_winner(model_set, T, learning_dict)
  fill_learning_dict(learning_dict, T, 'alarms', 1, param_is_separate = True)
  fill_learning_dict(learning_dict, T, 'EM_lens', 1, param_is_separate = True)
  fill_learning_dict(learning_dict, T, 'prominent_models', winner_model, param_is_separate = True)
  return learning_dict      
      
      
def evaluate_prominent(data, learning_dict, sigma_r, pp_thr, t, num_particles, Sigma_0_1task = None, Sigma_0_2task = None):
  if Sigma_0_1task is None:
    Sigma_0_1task = Sigma_0_1task_def
  if Sigma_0_2task is None:
    Sigma_0_2task = Sigma_0_2task_def
  prom_model = learning_dict['prominent_models'][-1]
  prev_posterior = learning_dict[prom_model]['posteriors'][-1]
  complexity = task_complexity(prom_model)
  if complexity == '1_task':
    mmllh_prev = learning_dict[prom_model]['mmllh'][0, t - 2]  # entire column is populated by the same value
    mmllh, posterior_new, pp = f.calc_mmllh_1task(data, sigma_r, prom_model, evaluation = "iterative", posterior = prev_posterior, mmllh_prev = mmllh_prev)
    if pp < pp_thr:
      alarm = 1
    else:
      alarm = 0
    learning_dict[prom_model]['mmllh'][:, t - 1] = mmllh
    if alarm:
      fill_learning_dict(learning_dict, t, 'posteriors', prev_posterior, prom_model)
    else:
      fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
  elif complexity == '2_task':
    if 'bg' in prom_model:
      prev_contexts = learning_dict[prom_model]['contexts'][-1]
      mmllhs_prev = learning_dict[prom_model]['mmllhs'][-1]
      mmllhs_new, posterior_new, contexts_new, pp = f.calc_mmllh_2task(data, sigma_r, prom_model.replace('_bg', ''),
                      evaluation = "iterative",
                      marginalize = False,
                      posterior_prev = prev_posterior,
                      mmllhs_prev = mmllhs_prev,
                      prev_contexts = prev_contexts)
      if pp < pp_thr:
        alarm = 1
      else:
        alarm = 0
      mmllh_acc = 1.
      for context in mmllhs_new:
        mmllh_acc *= mmllhs_new[context]
      learning_dict[prom_model]['mmllh'][:, t - 1] = mmllh_acc
      if alarm:
        fill_learning_dict(learning_dict, t, 'mmllhs', mmllhs_new, prom_model)
        fill_learning_dict(learning_dict, t, 'contexts', contexts_new, prom_model)
        fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
      else:
        fill_learning_dict(learning_dict, t, 'mmllhs', mmllhs_prev, prom_model)
        fill_learning_dict(learning_dict, t, 'contexts', prev_contexts, prom_model)
        fill_learning_dict(learning_dict, t, 'posteriors', prev_posterior, prom_model)
    else:
      mmllh_prev = learning_dict[prom_model]['mmllh'][0, t - 2]
      mmllh_new, posterior_new, pp = f.calc_mmllh_2task(data, sigma_r, prom_model, 
                      evaluation = "iterative", 
                      marginalize = True, 
                      posterior_prev = prev_posterior, 
                      mmllh_prev = mmllh_prev,
                      num_particles = num_particles)
      if pp < pp_thr:
        alarm = 1
      else:
        alarm = 0
      learning_dict[prom_model]['mmllh'][:, t - 1] = mmllh_new
      if alarm:
        fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
      else:
        fill_learning_dict(learning_dict, t, 'posteriors', prev_posterior, prom_model)
  # this function handles only the alarms list, evaluate_non_prominents() will take care of EM_lens list

  fill_learning_dict(learning_dict, t, 'alarms', alarm, param_is_separate = True)
  return learning_dict, alarm == 1


def evaluate_non_prominents(data, learning_dict, sigma_r, dream_idx, model_set, num_particles, t, D, Sigma_0_1task = None, Sigma_0_2task = None):
  if Sigma_0_1task is None:
    Sigma_0_1task = Sigma_0_1task_def
  if Sigma_0_2task is None:
    Sigma_0_2task = Sigma_0_2task_def
  prom_model = learning_dict['prominent_models'][-1]
  non_prom_models = deepcopy(model_set)
  non_prom_models.remove(prom_model)
  for model in non_prom_models:
    complexity = task_complexity(model)
    if complexity == '1_task':
      mmllh, posterior = f.calc_mmllh_1task(data, sigma_r, model, evaluation = "full")
      learning_dict[model]['mmllh'][dream_idx, t - 1] = mmllh
      if dream_idx == D - 1:  # posterior from the last dream is retained
        fill_learning_dict(learning_dict, t, 'posteriors', posterior, model)
    elif complexity == '2_task':
      if 'bg' in model:
        mmllhs, posterior, contexts = f.calc_mmllh_2task(data, sigma_r, model.replace('_bg', ''), marginalize = False, evaluation = "full")
        mmllh_acc = 1.
        for context in mmllhs:
          mmllh_acc *= mmllhs[context]
        mmllh = mmllh_acc
        if dream_idx == D - 1:  # last dream is retained
          fill_learning_dict(learning_dict, t, 'contexts', contexts, model)
          fill_learning_dict(learning_dict, t, 'mmllhs', mmllhs, model)
      else:
        mmllh, posterior = f.calc_mmllh_2task(data, sigma_r, model, num_particles = num_particles, evaluation = "full")
      learning_dict[model]['mmllh'][dream_idx, t - 1] = mmllh
      if dream_idx == D - 1:  # posterior from the last dream is retained
        fill_learning_dict(learning_dict, t, 'posteriors', posterior, model)

  if dream_idx == D - 1:
    learning_dict, winner_model = who_is_the_winner(model_set, t, learning_dict)
  return learning_dict


def who_is_the_winner(model_set, t, learning_dict = None):
  '''
  learning_dict records the evolution of mmllh, posterior, prominent model, pp, alarm and lenght of EM for some models
  mmllh is a numpy.ndarray of shape (D, -1)
  '''
  if learning_dict is None:
    learning_dict = dict()
  if learning_dict:
    winner_model = ''
    best_log_mmllh = -1e200
    for model in model_set:
      model_props = learning_dict[model]
      mean_of_log_mmllh = np.mean(np.log(model_props['mmllh']), axis = 0)
      current_log_mmllh = mean_of_log_mmllh[t - 1]
      if current_log_mmllh > best_log_mmllh:
        best_log_mmllh = current_log_mmllh
        winner_model = model
    
    len_of_winner_list = len(learning_dict['winning_models'])
    if len_of_winner_list < t - 1:
      len_diff = t - 1 - len_of_winner_list
      learning_dict['winning_models'].extend([None] * len_diff)
      learning_dict['winning_models'].append(winner_model)
    elif len_of_winner_list == t - 1:
      learning_dict['winning_models'].append(winner_model)
    else:
      raise Exception("length of winning list exceeds t - 1") 
    return learning_dict, winner_model
  else:
    raise Exception("learning_dict is empty")


def data_generator(data):
    n = size_of_data(data)
    z = data['z']
    r = data['r']
    c = data['c']
    for i in range(n):
      yield dict(z = np.array([z[i]]), r = np.array([r[i]]), c = [c[i]])

def init_learning_dict(model_set, D, T):
  learning_dict = dict()
  model_props = ['mmllh', 'posteriors', 'contexts', 'mmllhs']
  separate_props = ['prominent_models', 'winning_models', 'alarms', 'EM_lens']
  for model in model_set:
    model_props_copy = deepcopy(model_props)
    if 'bg' not in model and 'contexts' in model_props_copy:
      model_props_copy.remove('contexts')
      model_props_copy.remove('mmllhs')
    learning_dict[model] = dict()
    for prop in model_props_copy:
      if prop == "mmllh":
        learning_dict[model][prop] = np.zeros((D, T))
      else:
        learning_dict[model][prop] = []
  for prop in separate_props:
    learning_dict[prop] = []
  return learning_dict

def update_prominent_model_korabbi(learning_dict, t, new_point_is_exciting, EM_len, EM_size_limit):
  winning_model = learning_dict['winning_models'][-1]
  prev_prominent_model = learning_dict['prominent_models'][-1]
  if new_point_is_exciting and EM_len > EM_size_limit:
      fill_learning_dict(learning_dict, t, 'prominent_models', winning_model, param_is_separate = True)
  else:
    fill_learning_dict(learning_dict, t, 'prominent_models', prev_prominent_model, param_is_separate = True)

def update_prominent_model(learning_dict, t):
  winning_model = learning_dict['winning_models'][-1]
  fill_learning_dict(learning_dict, t, 'prominent_models', winning_model, param_is_separate = True)

def GR_EM_learner_korabbi(data, sigma_r, model_set, num_particles = 256, D = 10, pp_thr = .2, EM_size_limit = 1, verbose = False):
  T = size_of_data(data)
  data_gen = data_generator(data) # Python generator for iterating through data points 
  learning_dict = init_learning_dict(model_set, D, T)
  EM_exists = False
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  for idx, new_data in enumerate(data_gen):
    t = idx + 1
    if t == 1:  # evaluate all models "fully"
      learning_dict = evaluate_all_full(new_data, learning_dict, sigma_r, model_set, num_particles)
      prominent_model = learning_dict['prominent_models'][-1]
      if verbose:
        pbar.add(1)
    else:
      prominent_model_prev = prominent_model
      learning_dict, new_point_is_exciting = evaluate_prominent(new_data, learning_dict, sigma_r, pp_thr, t, num_particles)
      if new_point_is_exciting:
        if EM_exists:
          EM = h.concatenate_data([EM, new_data])
        else:
          EM = deepcopy(new_data)
          EM_exists = True
      if EM_exists:
        EM_len = size_of_data(EM)
        if 'bg' not in prominent_model:
          num_points_to_dream = t - EM_len
        else:
          num_points_to_dream = learning_dict[prominent_model]['posteriors'][-1][-1]
        fill_learning_dict(learning_dict, t, 'EM_lens', EM_len, param_is_separate = True)
      else:
        EM_len = 0
        if 'bg' not in prominent_model:
          num_points_to_dream = t
        else:
          num_points_to_dream = learning_dict[prominent_model]['posteriors'][-1][-1]
        fill_learning_dict(learning_dict, t, 'EM_lens', 0, param_is_separate = True)

      # GR D times
      for dream_idx in range(D):
        data_dream = h.GR(learning_dict, how_many = num_points_to_dream)
        if EM_exists:
          data_whole = h.concatenate_data([data_dream, EM])
        else:
          data_whole = data_dream
        learning_dict = evaluate_non_prominents(data_whole, learning_dict, sigma_r, dream_idx, model_set, num_particles, t, D)
      
      update_prominent_model(learning_dict, t, new_point_is_exciting, EM_len, EM_size_limit)
      prominent_model = learning_dict['prominent_models'][-1]
      if prominent_model != prominent_model_prev:
        EM_exists = False
      if verbose:
        pbar.add(1)
  return learning_dict


def GR_EM_learner(data, sigma_r, model_set, num_particles = 256, D = 10, pp_thr = .2, EM_size_limit = 0, verbose = False):
  T = size_of_data(data)
  data_gen = data_generator(data) # Python generator for iterating through data points 
  learning_dict = init_learning_dict(model_set, D, T)
  EM_exists = False
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  for idx, new_data in enumerate(data_gen):
    t = idx + 1
    if t == 1:  # evaluate all models "fully"
      FIRST_CONTEXT = new_data['c'][0]
      learning_dict = evaluate_all_full(new_data, learning_dict, sigma_r, model_set, num_particles)
      prominent_model = learning_dict['prominent_models'][-1]
      full_gt_data = new_data
      EM = deepcopy(new_data)
      EM_exists = True
      EM_len = 1
      print('prominent: ' + prominent_model)
      if verbose:
        pbar.add(1)
    else:
      prominent_model_prev = prominent_model
      learning_dict, new_point_is_exciting = evaluate_prominent(new_data, learning_dict, sigma_r, pp_thr, t, num_particles)
      full_gt_data = h.concatenate_data((full_gt_data, new_data))
      contexts = full_gt_data['c']
      if new_point_is_exciting and EM_exists and EM_len == EM_size_limit:
        new_point_is_exciting = False
      if new_point_is_exciting:
        if EM_exists:
            if prominent_model == 'x' or prominent_model == 'y':
                if len(np.unique(EM['c'])) == 2:
                    pass
                    #homogenize_EM(EM)
            EM = h.concatenate_data([EM, new_data])
        else:
          EM = deepcopy(new_data)
          EM_exists = True
      if EM_exists:
        EM_len = size_of_data(EM)
        if 'bg' not in prominent_model:
          num_points_to_dream = t - EM_len
        else:
          num_points_to_dream = learning_dict[prominent_model]['posteriors'][-1][-1]
          print('posterior')
          print(learning_dict[prominent_model]['posteriors'][-1])
        fill_learning_dict(learning_dict, t, 'EM_lens', EM_len, param_is_separate = True)
      else:
        EM_len = 0
        if 'bg' not in prominent_model:	
          num_points_to_dream = t
        else:
          num_points_to_dream = learning_dict[prominent_model]['posteriors'][-1][-1]
          print('posterior')
          print(learning_dict[prominent_model]['posteriors'][-1])
        fill_learning_dict(learning_dict, t, 'EM_lens', 0, param_is_separate = True)

      # GR D times
      for dream_idx in range(D):
        if num_points_to_dream:
            data_dream = h.GR(learning_dict, how_many = num_points_to_dream, first_context = FIRST_CONTEXT)
        if EM_exists:
          if num_points_to_dream:
              data_whole = h.concatenate_data([data_dream, EM])
          else:
              data_whole = EM
        else:
          data_whole = data_dream
        learning_dict = evaluate_non_prominents(data_whole, learning_dict, sigma_r, dream_idx, model_set, num_particles, t, D)
      if len(np.unique(contexts)) == 1:
        fill_learning_dict(learning_dict, t, 'prominent_models', prominent_model_prev, param_is_separate = True)
      else:
        update_prominent_model(learning_dict, t)
      prominent_model = learning_dict['prominent_models'][-1]
      print('prominent: ' + prominent_model)
      if prominent_model != prominent_model_prev:
        EM_exists = False
      if verbose:
        pbar.add(1)
  return learning_dict







  