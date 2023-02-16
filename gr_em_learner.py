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
    learning_dict[model][param_name][T - 1] = param_value
  else:
    learning_dict[param_name][T - 1] = param_value
      
def evaluate_all(data, learning_dict, sigma_r, model_set, num_particles, Sigma_0_1task = None, Sigma_0_2task = None):
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
        for context_ in mmllhs:
          mmllh *= mmllhs[context_]
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
  prom_model = learning_dict['prominent_models'][t - 2]
  prev_posterior = learning_dict[prom_model]['posteriors'][t - 2]
  complexity = task_complexity(prom_model)
  if complexity == '1_task':
    mmllh_prev = learning_dict[prom_model]['mmllh'][0, t - 2]  # entire column is populated by the same value
    mmllh, posterior_new, pp = f.calc_mmllh_1task(data, sigma_r, prom_model, evaluation = "iterative", posterior = prev_posterior, mmllh_prev = mmllh_prev)
    if pp < pp_thr:
      alarm = 1
    else:
      alarm = 0
    learning_dict[prom_model]['mmllh'][:, t - 1] = mmllh
    # new post, etc is filled in by all means and mmllh_test will decide on the need for replacing these values
    fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
  elif complexity == '2_task':
    if 'bg' in prom_model:
      prev_contexts = learning_dict[prom_model]['contexts'][t - 2]
      mmllhs_prev = learning_dict[prom_model]['mmllhs'][t - 2]
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
      # new post, etc is filled in by all means and mmllh_test will decide on the need for replacing these values
      fill_learning_dict(learning_dict, t, 'mmllhs', mmllhs_new, prom_model)
      fill_learning_dict(learning_dict, t, 'contexts', contexts_new, prom_model)
      fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
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
      # new post, etc is filled in by all means and mmllh_test will decide on the need for replacing these values
      fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
  fill_learning_dict(learning_dict, t, 'alarms', alarm, param_is_separate = True)
  return learning_dict, alarm == 1


def evaluate_non_prominents(data, learning_dict, sigma_r, dream_idx, model_set, num_particles, t, D, Sigma_0_1task = None, Sigma_0_2task = None):
  if Sigma_0_1task is None:
    Sigma_0_1task = Sigma_0_1task_def
  if Sigma_0_2task is None:
    Sigma_0_2task = Sigma_0_2task_def
  prom_model = learning_dict['prominent_models'][t - 2]
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

  model_change_is_necessary = 'not known'
  if dream_idx == D - 1:
    learning_dict, winner_model = who_is_the_winner(model_set, t, learning_dict)
    if prom_model != winner_model:
      model_change_is_necessary = True
    else:
      model_change_is_necessary = False
  return learning_dict, model_change_is_necessary


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
    
    learning_dict['winning_models'][t - 1] = winner_model
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
        learning_dict[model][prop] = [None] * T
  for prop in separate_props:
    learning_dict[prop] = [None] * T
  return learning_dict


def update_prominent_model(learning_dict, t, new_point_is_exciting, EM_full, model_change_is_necessary):
  winning_model = learning_dict['winning_models'][t - 1]
  prev_prominent_model = learning_dict['prominent_models'][t - 2]
  if new_point_is_exciting and model_change_is_necessary:
      fill_learning_dict(learning_dict, t, 'prominent_models', winning_model, param_is_separate = True)
  else:
    fill_learning_dict(learning_dict, t, 'prominent_models', prev_prominent_model, param_is_separate = True)


def update_prominent_model_old(learning_dict, t):
  winning_model = learning_dict['winning_models'][t - 1]
  fill_learning_dict(learning_dict, t, 'prominent_models', winning_model, param_is_separate = True)


def rewind_prominent_posterior(learning_dict, t, lag):
  prom_model = learning_dict['prominent_models'][t - 2]  # prom model in the (t-1). pos.
  lagged_posterior = learning_dict[prom_model]['posteriors'][t - 1 - lag]  # post. to which we want to rewind
  learning_dict[prom_model]['posteriors'][t - 1] = lagged_posterior


def mmllh_test(learning_dict, new_data, new_point_is_exciting, EM_full, num_points_to_dream, D, sigma_r, model_set, num_particles, t, first_context, pp_thr, EM = None):
  '''
  arranges the various branches in the decision tree (e.g. decides on the need for replacing the posterior of prom model)
  '''
  for dream_idx in range(D):
    if num_points_to_dream:
      data_dream = h.GR(learning_dict, t, how_many=num_points_to_dream, first_context=first_context)
    if EM is not None:
      if num_points_to_dream:
        data_whole = h.concatenate_data([data_dream, EM, new_data])
      else:
        data_whole = h.concatenate_data([EM, new_data])
    else:
      data_whole = data_dream
    learning_dict, model_change_is_necessary = evaluate_non_prominents(data_whole,
                                            learning_dict,
                                            sigma_r,
                                            dream_idx,
                                            model_set,
                                            num_particles,
                                            t,
                                            D)
  if new_point_is_exciting and not EM_full:
    rewind_prominent_posterior(learning_dict, t=t, lag=1)
  elif new_point_is_exciting and EM_full:
    # print('EM full, model change is necessary')
    if model_change_is_necessary:
      rewind_prominent_posterior(learning_dict, t=t, lag=1)
    else:
      # evaluate prominent: t. position, {new_data, EM}
      data = h.concatenate_data([new_data, EM])
      learning_dict, new_point_is_exciting = evaluate_prominent(data, learning_dict, sigma_r, pp_thr, t,
                                                                num_particles)
  # if new point is not exciting, nothing happens to the prominent posterior, it stays updated by the newest data point
  return model_change_is_necessary

def GR_EM_learner(data, sigma_r, model_set, num_particles = 256, D = 10, pp_thr = .2, EM_size_limit = 0, verbose = False):
  T = size_of_data(data)
  data_gen = data_generator(data) # Python generator for iterating through data points 
  learning_dict = init_learning_dict(model_set, D, T)
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  for idx, new_data in enumerate(data_gen):
    t = idx + 1
    print(t)
    if t == 1:  # evaluate all models, the first point wont be built in the post. of any model
      FIRST_CONTEXT = new_data['c'][0]
      learning_dict = evaluate_all(new_data, learning_dict, sigma_r, model_set, num_particles)
      prominent_model = learning_dict['prominent_models'][0]
      full_gt_data = new_data
      EM = deepcopy(new_data)
      EM_len = 1
      print('prominent: ' + prominent_model)
      if verbose:
        pbar.add(1)
    else:
      EM_full = (EM_len == EM_size_limit)
      EM_exists_and_not_full = (EM_len < EM_size_limit and EM_len > 0)
      EM_exists = EM_exists_and_not_full or EM_full
      prominent_model_prev = prominent_model
      learning_dict, new_point_is_exciting = evaluate_prominent(new_data, learning_dict, sigma_r, pp_thr, t, num_particles)
      full_gt_data = h.concatenate_data((full_gt_data, new_data))
      contexts = full_gt_data['c']
      if 'bg' not in prominent_model:
        num_points_to_dream = t - EM_len - 1  # new_data is observed
      else:
        num_points_to_dream = learning_dict[prominent_model]['posteriors'][t-1][-1]
      # the various branches differ from each other regarding what happens to EM essentially
      # mmllh_test under the hood takes care of dealing with prominent model in the appropriate way
      if EM_exists:
        model_change_is_necessary = mmllh_test(learning_dict, new_data, new_point_is_exciting, EM_full,
                                               num_points_to_dream, D, sigma_r,
                                               model_set, num_particles, t, FIRST_CONTEXT, pp_thr, EM=EM)
      else:
        model_change_is_necessary = mmllh_test(learning_dict, new_data, new_point_is_exciting, EM_full,
                                               num_points_to_dream, D, sigma_r,
                                               model_set, num_particles, t, FIRST_CONTEXT, pp_thr)
      if new_point_is_exciting:
        if EM_full:                                   # EM is cleared on this branch
          EM_len = 0
        elif EM_exists_and_not_full:                  # EM is augmented on this branch
          EM = h.concatenate_data([EM, new_data])
          EM_len = size_of_data(EM)
        else:                                         # EM is born on this branch
          EM = deepcopy(new_data)
          EM_len = 1
      else:                                           # EM remains the same on this branch
        pass

      fill_learning_dict(learning_dict, t, 'EM_lens', EM_len, param_is_separate=True)
      if len(np.unique(contexts)) == 1:
        model_change_is_necessary = False
        fill_learning_dict(learning_dict, t, 'prominent_models', prominent_model_prev, param_is_separate = True)
      else:
        update_prominent_model(learning_dict, t, new_point_is_exciting, EM_full, model_change_is_necessary)
      prominent_model = learning_dict['prominent_models'][t-1]
      print('prominent: ' + prominent_model)
      if new_point_is_exciting and model_change_is_necessary:  # EM is cleared
        EM_len = 0
      if verbose:
        pbar.add(1)
  return learning_dict







  