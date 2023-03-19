import sys
import functions_for_2task as f
import helper_mate as h
from helper_mate import size
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
      
def evaluate_all(data, learning_dict, sigma_r, model_set, num_particles):
  T = size(data)
  context = data['c'][0]
  posterior_for_each_model = dict(zip(model_set, [None] * len(model_set)))
  for model in model_set:
    complexity = task_complexity(model)
    if complexity == '1_task':
      mmllh, post = f.calc_mmllh_1task(data, sigma_r, model, evaluation = "full")
      posterior_for_each_model[model] = post
      learning_dict[model]['mmllh'][:, T - 1] = mmllh
    elif complexity == '2_task':
      if 'bg' in model:
        mmllhs, post, contexts = f.calc_mmllh_2task(data, sigma_r, model.replace('_bg', ''), marginalize = False, evaluation = "full")
        posterior_for_each_model[model] = post
        mmllh = 1.
        for context_ in mmllhs:
          mmllh *= mmllhs[context_]
        fill_learning_dict(learning_dict, T, 'contexts', contexts, model)
        fill_learning_dict(learning_dict, T, 'mmllhs', mmllhs, model)
      else:
        mmllh, post = f.calc_mmllh_2task(data, sigma_r, model, num_particles = num_particles, evaluation = "full")
        posterior_for_each_model[model] = post
      learning_dict[model]['mmllh'][:, T - 1] = mmllh
  learning_dict, winner_model = who_is_the_winner(model_set, T, learning_dict)
  fill_learning_dict(learning_dict, T, 'prominent_models', winner_model, param_is_separate = True)
  for model in model_set:
    fill_learning_dict(learning_dict, T, 'posteriors', posterior_for_each_model[model], model)
  return learning_dict

def find_last_idx_of_eval(a):
  a = np.array(a)
  return np.max(np.where(a != None))

def is_surprising(data, learning_dict, sigma_r, pp_thr, t, num_particles):
  prom_model = learning_dict['prominent_models'][t - 2]
  last_idx_of_eval = find_last_idx_of_eval(learning_dict[prom_model]['mmllh'])
  prev_posterior = learning_dict[prom_model]['posteriors'][t - 2]
  complexity = task_complexity(prom_model)
  if complexity == '1_task':
    mmllh_prev = learning_dict[prom_model]['mmllh'][0, t - 2]  # entire column is populated by the same value
    mmllh, posterior_new, pp = f.calc_mmllh_1task(data, sigma_r, prom_model, evaluation="iterative",
                                                  posterior=prev_posterior, mmllh_prev=mmllh_prev)
    if pp < pp_thr:
      alarm = 1
    else:
      alarm = 0
    learning_dict[prom_model]['mmllh'][:, t - 1] = mmllh
    # fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
    fill_learning_dict(learning_dict, t, 'alarms', alarm, param_is_separate=True)
    return learning_dict, alarm == 1, posterior_new
  elif complexity == '2_task':
    if 'bg' in prom_model:
      prev_contexts = learning_dict[prom_model]['contexts'][t - 2]
      mmllhs_prev = learning_dict[prom_model]['mmllhs'][t - 2]
      mmllhs_new, posterior_new, contexts_new, pp = f.calc_mmllh_2task(data, sigma_r, prom_model.replace('_bg', ''),
                                                                       evaluation="iterative",
                                                                       marginalize=False,
                                                                       posterior_prev=prev_posterior,
                                                                       mmllhs_prev=mmllhs_prev,
                                                                       prev_contexts=prev_contexts)
      if pp < pp_thr:
        alarm = 1
      else:
        alarm = 0
      mmllh_acc = 1.
      for context in mmllhs_new:
        mmllh_acc *= mmllhs_new[context]
      learning_dict[prom_model]['mmllh'][:, t - 1] = mmllh_acc
      # fill_learning_dict(learning_dict, t, 'mmllhs', mmllhs_new, prom_model)
      # fill_learning_dict(learning_dict, t, 'mmllhs_bound_to_posterior', mmllhs_new, prom_model)
      # fill_learning_dict(learning_dict, t, 'contexts', contexts_new, prom_model)
      # fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
      return learning_dict, alarm == 1, posterior_new, contexts_new, mmllhs_new
    else:
      mmllh_prev = learning_dict[prom_model]['mmllh'][0, t - 2]
      mmllh_new, posterior_new, pp = f.calc_mmllh_2task(data, sigma_r, prom_model,
                                                        evaluation="iterative",
                                                        marginalize=True,
                                                        posterior_prev=prev_posterior,
                                                        mmllh_prev=mmllh_prev,
                                                        num_particles=num_particles)
      if pp < pp_thr:
        alarm = 1
      else:
        alarm = 0
      learning_dict[prom_model]['mmllh'][:, t - 1] = mmllh_new
      # new post, etc is filled in by all means and mmllh_test will decide on the need for replacing these values
      # fill_learning_dict(learning_dict, t, 'posteriors', posterior_new, prom_model)
      return learning_dict, alarm == 1, posterior_new
      
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
      # fill_learning_dict(learning_dict, t, 'mmllhs_bound_to_posterior', mmllhs_new, prom_model)
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
  # fill_learning_dict(learning_dict, t, 'alarms', alarm, param_is_separate = True)
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
          # fill_learning_dict(learning_dict, t, 'mmllhs_bound_to_posterior', mmllhs, model)
      else:
        mmllh, posterior = f.calc_mmllh_2task(data, sigma_r, model, num_particles = num_particles, evaluation = "full")
      learning_dict[model]['mmllh'][dream_idx, t - 1] = mmllh
      if dream_idx == D - 1:  # posterior from the last dream is retained
        fill_learning_dict(learning_dict, t, 'posteriors', posterior, model)

  model_change_is_necessary = 'not known'
  if dream_idx == D - 1:
    learning_dict, winner_model = who_is_the_winner(model_set, t, learning_dict)
    fill_learning_dict(learning_dict, t, 'winning_models', winner_model, param_is_separate=True)
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
    n = size(data)
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
      #model_props_copy.remove('mmllhs_bound_to_posterior')
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


def rewind_quantity_related_to_prominent_model(learning_dict, quantity_to_rewind, t, lag):
  prom_model = learning_dict['prominent_models'][t - 2]  # prom model in the (t-1). pos.
  lagged_quantity = learning_dict[prom_model][quantity_to_rewind][t - 1 - lag]  # post. to which we want to rewind
  learning_dict[prom_model][quantity_to_rewind][t - 1] = lagged_quantity


def semanticise(learning_dict, new_data, num_points_to_dream, prominent_model, D, sigma_r, model_set, num_particles, t, first_context, pp_thr, EM = None):
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
    if model_change_is_necessary:
      winning_model = learning_dict['winning_models'][t - 1]
      fill_learning_dict(learning_dict, t, 'winning_models', winning_model, param_is_separate=True)
      fill_learning_dict(learning_dict, t, 'prominent_models', winning_model, param_is_separate=True)
    else:
      # evaluate prominent: t. position, {new_data, EM}
      data = h.concatenate_data([new_data, EM])
      learning_dict, _ = evaluate_prominent(data, learning_dict, sigma_r, pp_thr, t,
                                                                num_particles)
      fill_learning_dict(learning_dict, t, 'prominent_models', prominent_model, param_is_separate=True)
      fill_learning_dict(learning_dict, t, 'winning_models', prominent_model, param_is_separate=True)
  return learning_dict


def mmllh_test(learning_dict, new_data, new_point_is_exciting, EM_full, num_points_to_dream, D, sigma_r, model_set, num_particles, t, first_context, pp_thr, prom_model, EM = None):
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
    pass
    # rewind_quantity_related_to_prominent_model(learning_dict, 'posteriors', t = t, lag = 1)
    if 'bg' in prom_model:
      pass
      # rewind_quantity_related_to_prominent_model(learning_dict, 'contexts', t = t, lag = 1)
      # rewind_quantity_related_to_prominent_model(learning_dict, 'mmllhs_bound_to_posterior', t=t, lag=1)
  elif new_point_is_exciting and EM_full:
    # print('EM full, model change is necessary')
    if model_change_is_necessary:
      rewind_quantity_related_to_prominent_model(learning_dict, 'posteriors', t=t, lag=1)
      if 'bg' in prom_model:
        rewind_quantity_related_to_prominent_model(learning_dict, 'contexts', t=t, lag=1)
        rewind_quantity_related_to_prominent_model(learning_dict, 'mmllhs_bound_to_posterior', t=t, lag=1)
    else:
      # evaluate prominent: t. position, {new_data, EM}
      data = h.concatenate_data([new_data, EM])
      learning_dict, new_point_is_exciting = evaluate_prominent(data, learning_dict, sigma_r, pp_thr, t,
                                                                num_particles)
  # if new point is not exciting, nothing happens to the prominent posterior, it stays updated by the newest data point
  return learning_dict, model_change_is_necessary

def pass_along_previous_prominent_data(learning_dict, t, prominent_model):
  prev_prom_post = learning_dict[prominent_model]['posteriors'][t - 2]
  fill_learning_dict(learning_dict, t, 'posteriors', prev_prom_post, prominent_model)
  if 'bg' in prominent_model:
    mmllhs_prev = learning_dict[prominent_model]['mmllhs'][t - 2]
    contexts_prev = learning_dict[prominent_model]['contexts'][t - 2]
    fill_learning_dict(learning_dict, t, 'mmllhs', mmllhs_prev, prominent_model)
    fill_learning_dict(learning_dict, t, 'contexts', contexts_prev, prominent_model)

def update_current_prominent_model(learning_dict, t, prominent_model, updated_data):
  if 'bg' in prominent_model:
    updated_prominent_posterior, mmllhs_new, contexts_new = updated_data
    fill_learning_dict(learning_dict, t, 'posteriors', updated_prominent_posterior, prominent_model)
    fill_learning_dict(learning_dict, t, 'mmllhs', mmllhs_new, prominent_model)
    fill_learning_dict(learning_dict, t, 'contexts', contexts_new, prominent_model)
  else:
    updated_prominent_posterior = updated_data
    fill_learning_dict(learning_dict, t, 'posteriors', updated_prominent_posterior, prominent_model)

def GR_EM_learner(data, sigma_r, model_set, EM_size_limit_for_eval, num_particles = 256, D = 10, pp_thr = .2, EM_size_limit = 0, verbose = False):
  assert EM_size_limit_for_eval <= EM_size_limit, "EM_size_limit_for_eval must be less than or equal to EM_size_limit"
  T = size(data)
  data_gen = data_generator(data)
  learning_dict = init_learning_dict(model_set, D, T)
  if verbose:
    pbar = tf.keras.utils.Progbar(T)
  cold_run = True
  EM_exists = False
  for idx, new_data in enumerate(data_gen):
    t = idx + 1
    if t == 1:
      first_context = new_data['c'][0]
    if cold_run:
      if not EM_exists:
        EM = deepcopy(new_data)
        EM_exists = True
      else:
        EM = h.concatenate_data([EM, new_data])
      EM_len = size(EM)
      fill_learning_dict(learning_dict, t, 'alarms', 1, param_is_separate=True)
      fill_learning_dict(learning_dict, t, 'EM_lens', EM_len, param_is_separate=True)
      if EM_len == EM_size_limit_for_eval:
        cold_run = False
        learning_dict = evaluate_all(EM, learning_dict, sigma_r, model_set, num_particles)
        EM_len = 0
        prominent_model = learning_dict['prominent_models'][t - 1]
        fill_learning_dict(learning_dict, t, 'prominent_models', prominent_model, param_is_separate=True)

    else:
      EM_full = (EM_len == EM_size_limit)
      EM_exists_and_not_full = (EM_len < EM_size_limit and EM_len > 0)
      EM_exists = EM_exists_and_not_full or EM_full
      EM_is_big_enough_to_eval = (EM_len == EM_size_limit_for_eval)
      # SURPRISINGNESS
      if 'bg' in prominent_model:
        learning_dict, new_point_is_surprising, updated_prominent_posterior, contexts_new, mmllhs_new = \
          is_surprising(new_data, learning_dict, sigma_r, pp_thr, t, num_particles)
      else:
        learning_dict, new_point_is_surprising, updated_prominent_posterior = \
          is_surprising(new_data, learning_dict, sigma_r, pp_thr, t, num_particles)

      if new_point_is_surprising:
        if EM_full or EM_is_big_enough_to_eval:
          # how many points should be dreamed?
          if 'bg' not in prominent_model:
            num_points_to_dream = t - EM_len - 1
          else:
            data_p_counter_list = learning_dict[prominent_model]['posteriors'][t - 2][-1]
            need_for_dream_indicator = t - EM_len - 1
            if need_for_dream_indicator:
              num_points_to_dream = data_p_counter_list
            else:
              num_points_to_dream = dict()
          learning_dict = semanticise(learning_dict, new_data, num_points_to_dream, prominent_model, D, sigma_r,
                      model_set, num_particles, t, first_context, pp_thr, EM=EM)
          EM_len = 0
          fill_learning_dict(learning_dict, t, 'alarms', 1, param_is_separate=True)
          fill_learning_dict(learning_dict, t, 'EM_lens', EM_len, param_is_separate=True)
        else:  # prom doesnt change
          if EM_exists:
            EM = h.concatenate_data([EM, new_data])
          else:
            EM = deepcopy(new_data)
          EM_len = size(EM)
          pass_along_previous_prominent_data(learning_dict, t, prominent_model)
          fill_learning_dict(learning_dict, t, 'prominent_models', prominent_model, param_is_separate=True)
          fill_learning_dict(learning_dict, t, 'alarms', 1, param_is_separate=True)
          fill_learning_dict(learning_dict, t, 'EM_lens', EM_len, param_is_separate=True)
      else:  # update current prominent, prom model doesnt change
        if 'bg' in prominent_model:
          updated_data = updated_prominent_posterior, mmllhs_new, contexts_new
        else:
          updated_data = updated_prominent_posterior
        EM_len = size(EM)
        update_current_prominent_model(learning_dict, t, prominent_model, updated_data)
        fill_learning_dict(learning_dict, t, 'prominent_models', prominent_model, param_is_separate=True)
        fill_learning_dict(learning_dict, t, 'alarms', 0, param_is_separate=True)
        fill_learning_dict(learning_dict, t, 'EM_lens', EM_len, param_is_separate=True)
      prominent_model = learning_dict['prominent_models'][t - 1]

  return learning_dict

if __name__ == '__main__':
  # Data parameters
  from gt_learner import GT_learner
  SCHEDULE = 'BLOCKED'  # 'BLOCKED' or 'INTERLEAVED' or 'CUSTOM'

  BLOCK_SIZE = 4  # only applies if SCHEDULE is 'CUSTOM'
  N_BATCHES = 1  # only applies if SCHEDULE is 'CUSTOM'
  T = N_BATCHES * 2 * BLOCK_SIZE  # only applies if SCHEDULE is 'BLOCKED' or 'INTERLEAVED
  ALPHA_LIST = [45, -45]
  N_RUNS = 1

  # Agent parameters
  SIGMA_R = .3
  PP_THRESHOLD = 100.2
  D = 5
  EM_SIZE = 4
  EM_size_limit_for_eval = 4
  # Generate N_RUNS datasets
  datasets = [h.generate_batch_data(ALPHA_LIST, BLOCK_SIZE, N_BATCHES) for i in range(N_RUNS)]

  # Define models to be tested
  model_set = ['x', 'y', '1x2D', '2x2D_bg']
  data = datasets[0]
  result = GR_EM_learner(data, SIGMA_R, model_set, EM_size_limit_for_eval, verbose=False,
                               EM_size_limit=EM_SIZE, pp_thr=PP_THRESHOLD, D=D)
  # result_gt = GT_learner(data, SIGMA_R, model_set)
  a = 1




  