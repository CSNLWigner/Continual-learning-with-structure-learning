from copy import deepcopy
import functions_for_2task as f
from gr_em_learner import who_is_the_winner, data_generator, fill_learning_dict, task_complexity, size_of_data
import numpy as np

def init_learning_dict_GT(model_set, T):
  learning_dict = dict()
  model_props = ['mmllh']
  separate_props = ['prominent_models', 'winning_models']
  for model in model_set:
    model_props_copy = deepcopy(model_props)
    learning_dict[model] = dict()
    for prop in model_props_copy:
      if prop == "mmllh":
        learning_dict[model][prop] = np.zeros((1, T))
      else:
        learning_dict[model][prop] = [None] * T
  for prop in separate_props:
    learning_dict[prop] = [None] * T
  return learning_dict
  
def GT_learner(data, sigma_r, model_set, num_particles = 256):
  T = size_of_data(data)
  learning_dict = init_learning_dict_GT(model_set, T)
  for model in model_set:
    data_gen = data_generator(data) # Python generator for iterating through data points 
    complexity = task_complexity(model)
    if complexity == '1_task':
      mmllh_list, _ = f.calc_mmllh_1task(data, sigma_r, model, evaluation = "full", ret_all_mmllhs = True)
      learning_dict[model]['mmllh'][0, :] = mmllh_list
    elif complexity == '2_task':
      if 'bg' in model:
        mmllh_list = []
        for idx, new_data in enumerate(data_gen):
          if idx == 0:
            mmllhs, posterior, contexts = f.calc_mmllh_2task(new_data, sigma_r, model.replace('_bg', ''), marginalize = False, evaluation = "full")
            mmllh_acc = 1.
            for _, mmllh_ in mmllhs.items():
              mmllh_acc *= mmllh_
            mmllh_list.append(mmllh_acc)
          else:
            mmllhs, posterior, contexts, _ = f.calc_mmllh_2task(new_data, sigma_r, model.replace('_bg', ''), 
                                                              evaluation = "iterative", 
                                                              marginalize = False, 
                                                              posterior_prev = posterior, 
                                                              mmllhs_prev = mmllhs, 
                                                              prev_contexts = contexts)
            mmllh_acc = 1.
            for _, mmllh_ in mmllhs.items():
              mmllh_acc *= mmllh_
            mmllh_list.append(mmllh_acc)
        learning_dict[model]['mmllh'][0, :] = mmllh_list
      else:
        mmllh_list, _ = f.calc_mmllh_2task(data, sigma_r, model, num_particles = num_particles, evaluation = "full", marginalize = True, ret_all_mmllhs = True)
        learning_dict[model]['mmllh'][0, :] = mmllh_list
  contexts = data['c']
  for t in range(1, T + 1):
    if t == 1:
      learning_dict, winner_model = who_is_the_winner(model_set, t, learning_dict)
      fill_learning_dict(learning_dict, t, 'prominent_models', winner_model, param_is_separate = True)
    else:
      winner_prev = winner_model
      first_t_context = contexts[:t]
      if len(np.unique(first_t_context)) == 1:
        fill_learning_dict(learning_dict, t, 'prominent_models', winner_prev, param_is_separate = True)
      else:
        learning_dict, winner_model = who_is_the_winner(model_set, t, learning_dict)
        fill_learning_dict(learning_dict, t, 'prominent_models', winner_model, param_is_separate = True)
  return learning_dict  






