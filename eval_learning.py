import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import zip_longest
import helper
from functions_for_2task import calc_mmllh_1task
from functions_for_2task import calc_mmllh_2task
from copy import deepcopy

def learnGT(data,sigma_r,model_set=["x", "y"],marginalize=False, T0 = 0):
    '''
    Evaluates the evolution of the mmllh for a ground truth learner, which has access to the data throughout the lerarning.
    The function can evaluate five different models, the set of evaluated models can be specified by a parameter. 
    Model marginal likelihoods are calculated at each time step.

    Parameters
    ----------
    data :  dict
        dictionary containing all the information needed for completely define the dataset
        keys : 'z', 'r', 'c'
        values : np.ndarray (latent vectors), tensorflow.Tensor (rewards), list (contexts)
    sigma_r : float
    model_set : list of strings
        its value can be "x", "y", "1x2D", "2x1D", or "2x2D" 
    T0: first data point in the data set to evaluate, default = 0
    
    Returns
    ----------
    mmllh : 5xT float array, where T is the length of data
        5 lines correspond to the five potential models (IDs are standardized and are specified in the modelIDs variable). 
        Only those rows are filled that the model_set input variable sets for evaluation
    pred_prob : 5xT float array, the predictive probabilities of the different models based on their respective posteriors
    '''

    modelIDs={"x":0, "y":1, "1x2D":2, "2x1D":3, "2x2D":4}
        
    T = np.size(data["z"],0)
    
    #pbar = tf.keras.utils.Progbar(T)

    mllh=np.zeros([5, T])
    mus=[0,0,0,0,0]
    Sigmas=[0,0,0,0,0]
    alphas=[0,0,0,0,0]
    data_point_counter_list= [0,0,0,0,0]
    pred_prob=np.zeros([5, T])
    
    zs = data["z"]
    rs = data["r"]
    cs = data["c"]
    for modelID in model_set:
        #pbar.add(1)
        #print(modelID)
        mi = modelIDs.get(modelID)
        if modelID == "1x2D":
            Sigma_0 = [[1,0],[0,1]]
        else:
            Sigma_0 = 1
        for i in range(T0, T):
            dAct = {"z": np.zeros([i,2]), "r": np.zeros(i), "c": np.zeros(i)}
            dAct["z"]=zs[0:i+1:1]
            dAct["r"]=rs[0:i+1:1]
            dAct["c"]=cs[0:i+1:1]
            dNew={"z":np.zeros([1,2]), "r":np.zeros(1), "c":np.zeros(1)}
            dNew["z"][0]=zs[i]
            dNew["r"][0]=rs[i]
            dNew["c"][0]=cs[i]
            if modelID == "2x1D" or modelID == "2x2D":
                if marginalize:
                    if i==T0:
                        mllh[mi][i], posteriorPrev = calc_mmllh_2task(dAct, sigma_r, evaluation = 'full', model = modelID, num_particles = 264, marginalize = marginalize)
                    else:
                        mllh[mi][i], posteriorPrev = calc_mmllh_2task(dNew, sigma_r, evaluation = 'iterative', model = modelID, num_particles = 264, marginalize = marginalize, posterior_prev = posteriorPrev, mmllh_prev = mllh[mi][i-1])
                else:
                    if modelID == "2x1D":
                        mllh[mi][i], posteriorPrev = calc_mmllh_2task(dAct, sigma_r, evaluation = 'full', model = modelID, num_particles = 264, marginalize = marginalize)
                    else:
                        mllhs, posteriorPrev, contexts = calc_mmllh_2task(dAct, sigma_r, evaluation = 'full', model = modelID, num_particles = 264, marginalize = marginalize)
                        mllh[mi][i] = mllhs[0] * mllhs[1]
                # THE BELOW PART WILL BE IMPORTANT FOR THE ITERATIVE APPROACH
                #if marginalize == False:
                    #mus[mi], Sigmas[mi], data_point_counter_list[mi] = posterior2
                #else:
                    #mus[mi], Sigmas[mi], alphas[mi], data_point_counter_list[mi] = posterior2
            else:
                if i==T0:
                    mllh[mi][i], posteriorPrev = calc_mmllh_1task(dAct, sigma_r, model = modelID, evaluation = 'full')
                else:
                    mllh[mi][i], posteriorPrev, pred_prob[mi][i] = calc_mmllh_1task(dNew, sigma_r, model = modelID, evaluation = 'iterative', posterior = posteriorPrev, mmllh_prev = mllh[mi][i-1])
    return mllh, pred_prob

def dream_data_from_posterior(model, T = None, mu = None, Sigma = None,
                                mus = None, Sigmas = None, normalized_weights = None, data_point_counter_list = None, infness = 'non-inf'):
  '''
  1 tasknal mu es Sigma kell, 2 tasknal mus, Sigmas, meg a tobbi
  '''
  if model == 'x':
    post = tfd.Normal(loc = mu, scale = Sigma)
    gamma = np.array(post.sample(T))
    gamma_out = np.zeros((gamma.shape[0], 2))
    gamma_out[:, 1] = gamma
    data_dream = helper.generate_data_from_gammas(gamma_out, T)
    data_dream["c"]=np.ones(T)*90
    return data_dream
  elif model == 'y':
    post = tfd.Normal(loc = mu, scale = Sigma)
    gamma = np.array(post.sample(T))
    gamma_out = np.zeros((gamma.shape[0], 2))
    gamma_out[:, 0] = gamma
    data_dream = helper.generate_data_from_gammas(gamma_out, T)
    data_dream["c"]=np.ones(T)*0
    return data_dream
  elif model == '1x2D':
    post = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = Sigma)
    gamma_out = np.array(post.sample(T))
    data_dream = helper.generate_data_from_gammas(gamma_out, T)
    data_dream["c"]=np.ones(T)*45
    return data_dream
  elif model == '2x1D':
    bernoulli = tfd.Categorical(probs = normalized_weights)
    chosen_particle_idx = bernoulli.sample(1)
    Tx = data_point_counter_list[int(chosen_particle_idx)][0]
    Ty = data_point_counter_list[int(chosen_particle_idx)][1]
    
    #kulon x-re
    components_x = []
    for i in range(len(normalized_weights)):
      components_x.append(tfd.Normal(loc = np.float64(mus[i][0]), scale=np.float64(Sigmas[i][0])))
    post_x = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_x)
    gamma_ = np.array(post_x.sample(Tx))
    gammax = np.zeros((gamma_.shape[0], 2))
    gammax[:, 1] = gamma_

    #kulon y-ra
    components_y = []
    for i in range(len(normalized_weights)):
      components_y.append(tfd.Normal(loc = np.float64(mus[i][1]), scale=np.float64(Sigmas[i][1])))
    post_y = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_y)
    gamma_ = np.array(post_y.sample(Ty))
    gammay = np.zeros((gamma_.shape[0], 2))
    gammay[:, 0] = gamma_

    data_dream_x = helper.generate_data_from_gammas(gammax, Tx, infness = infness)
    data_dream_y = helper.generate_data_from_gammas(gammay, Ty, infness = infness)
    data_dream = helper.concatenate_data(data_dream_x, data_dream_y)
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
    

    #kulon y-ra
    components_y = []
    for i in range(len(normalized_weights)):
      components_y.append(tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[i][1]), covariance_matrix=np.float64(Sigmas[i][1])))
    post_y = tfd.Mixture(cat = tfd.Categorical(probs = normalized_weights), components = components_y)
      
    gammay = np.array(post_y.sample(Ty))
    
    data_dream_x = helper.generate_data_from_gammas(gammax, Tx, infness = infness)
    data_dream_y = helper.generate_data_from_gammas(gammay, Ty, infness = infness)

    data_dream_x["c"]=np.ones(T)*45
    data_dream_y["c"]=np.ones(T)*-45

    data_dream = helper.concatenate_data(data_dream_x, data_dream_y)
    
    return data_dream
  elif model == '2x1D_bg':
    #ilyenkor mus, Sigmas, data_point_counter_list: mus = [mu_x, mu_y]; Sigmas = [Sigma_x, Sigma_y]; data_point_counter_list = [Tx, Ty]

    Tx = data_point_counter_list[0]
    Ty = data_point_counter_list[1]
    #indices = list(np.arange(Tx + Ty))
    #y_indices = [item for item in indices if item not in x_indices]
    
    # almodas x-bol
    post_x = tfd.Normal(loc = np.float64(mus[0]), scale=np.float64(Sigmas[0]))
      
    gamma_ = np.array(post_x.sample(Tx))
    gammax = np.zeros((gamma_.shape[0], 2))
    gammax[:, 1] = gamma_

    # almodas y-bol
    post_y = tfd.Normal(loc = np.float64(mus[1]), scale=np.float64(Sigmas[1]))
     
    gamma_ = np.array(post_y.sample(Ty))
    gammay = np.zeros((gamma_.shape[0], 2))
    gammay[:, 0] = gamma_

    # shuffling
    data_dream_x = helper.generate_data_from_gammas(gammax, Tx)
    zx = data_dream_x['z']
    rx = data_dream_x['r']
    data_dream_y = helper.generate_data_from_gammas(gammay, Ty)
    zy = data_dream_y['z']
    ry = data_dream_y['r']
    data_dream = helper.concatenate_data(data_dream_x, data_dream_y)
    '''
    z = data_dream['z']
    r = data_dream['r']
    z[x_indices] = zx
    r[x_indices] = rx
    z[y_indices] = zy
    r[y_indices] = ry
    data_dream = {'z':z, 'r':r}
    '''   
    return data_dream
  elif model == '2x2D_bg':
    #ilyenkor mus, Sigmas, data_point_counter_list: mus = [mu_x, mu_y]; Sigmas = [Sigma_x, Sigma_y]; data_point_counter_list = [Tx, Ty]
    Tx = data_point_counter_list[0]
    Ty = data_point_counter_list[1]
    #indices = list(np.arange(Tx + Ty))
    #y_indices = [item for item in indices if item not in x_indices]
    
    # almodas x-bol
    post_x = tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[0]), covariance_matrix=np.float64(Sigmas[0]))
    
    gammax = np.array(post_x.sample(Tx))
    #gammax = np.zeros((gamma_.shape[0], 2))
    #gammax[:, 1] = gamma_

    # almodas y-bol
    post_y = tfd.MultivariateNormalFullCovariance(loc = np.float64(mus[1]), covariance_matrix=np.float64(Sigmas[1]))
     
    gammay = np.array(post_y.sample(Ty))
    #gammay = np.zeros((gamma_.shape[0], 2))
    #gammay[:, 0] = gamma_

    # shuffling
    data_dream_x = helper.generate_data_from_gammas(gammax, Tx)
    zx = data_dream_x['z']
    rx = data_dream_x['r']
    data_dream_y = helper.generate_data_from_gammas(gammay, Ty)
    zy = data_dream_y['z']
    ry = data_dream_y['r']
    data_dream = helper.concatenate_data(data_dream_x, data_dream_y)
    '''
    z = data_dream['z']
    r = data_dream['r']
    z[x_indices] = zx
    r[x_indices] = rx
    z[y_indices] = zy
    r[y_indices] = ry
    data_dream = {'z':z, 'r':r} 
    '''  
    return data_dream

def learnGR(data, grRepetition, sigma_r,model_set={"x", "y"}, marginalize=True):
    '''
    This is a test function to perform generative replay learning.
    The goal would be to iteratively evaluate a data set, and at every time 
    step only the posterior of the winning models si retained. In consecutive
    time steps the mllh of laternative models is evaluated through generative 
    replay from the actual model's posterior. 2-task models' mllh is calculated
    by marginalization over the background (recommended)

    The set of competing models is defined by the model_set list

    CURRENT CONSTRAINTS:
    * 2-task models are not properly evaluated
    * the function generating the GR data should be updated to use a posterior
    variable only that is a list of all the posterior components
    * the 1-task mllh calculating function should output a posterior instead of
    separate mu and Sigma
    * the evaluation of mllhs of multiple GR data sets should be carefully evaluated
    * currently the posterior corresponding to the highest mllh GR is retained after model switch
    '''

    competingModelNo = len(model_set)
    
    zs = data["z"]
    rs = data["r"]
    cs = data["c"]

    T = np.size(data["z"],0)

    mllh = np.zeros([competingModelNo,T])
    posteriors = [None] * 5
    winningModel= [None] * T

    t = 0
    dNew={"z":np.zeros([1,2]), "r":np.zeros(1), "c":np.zeros(1)}
    dNew["z"][0]=zs[t]
    dNew["r"][0]=rs[t]
    dNew["c"][0]=cs[t]
    
    for modelID in model_set:
        mi = model_set.index(modelID)
        if modelID=="x" or modelID=="y" or modelID=="1x2D":
            mllh[mi][t], posteriors[mi] = calc_mmllh_1task(dNew, sigma_r, model=modelID, evaluation = "full")
        else:
            mllh[mi,t], posteriors[mi] = calc_mmllh_2task(dNew, sigma_r, model=modelID, evaluation = "full", num_particles = 256, marginalize = True)
    winningModel[t] = model_set[np.argmax(mllh[:,t])]

    for t in range(1, T):
        #dAct = {"z": np.zeros([t,2]), "r": np.zeros(t), "c": np.zeros(t)}
        #dAct["z"]=zs[0:t+1:1]
        #dAct["r"]=rs[0:t+1:1]
        #dAct["c"]=cs[0:t+1:1]
        dNew={"z":np.zeros([1,2]), "r":np.zeros(1), "c":np.zeros(1)}
        dNew["z"][0]=zs[t]
        dNew["r"][0]=rs[t]
        dNew["c"][0]=cs[t]
        
        activeModel = winningModel[t-1]
        print(activeModel)
        
        dataGR = [None] * grRepetition
        mllhGR=np.zeros([competingModelNo, grRepetition])
        posteriorGR = [None] * grRepetition

        # GENERATING DATA FROM THE POSTERIOR OF THE ACTUAL MODEL
        # the same generated data set for all competing models
        for i in range(0,grRepetition):
            if activeModel=="x" or activeModel=="y" or activeModel=="1x2D":
                muAct = posteriors[model_set.index(winningModel[t-1])][0]
                SigmaAct = posteriors[model_set.index(winningModel[t-1])][1]
                dataGR[i] = dream_data_from_posterior(model = activeModel, T=t, mu = muAct, Sigma = SigmaAct)
            else:
                muAct = posteriors[model_set.index(winningModel[t-1])][0]
                SigmaAct = posteriors[model_set.index(winningModel[t-1])][1]
                normalized_weightsAct = posteriors[model_set.index(winningModel[t-1])][2]
                data_point_counter_listAct = posteriors[model_set.index(winningModel[t-1])][3]
                print(normalized_weightsAct)
                dataGR[i] = dream_data_from_posterior(model = activeModel, T=t, mu = muAct, Sigma = SigmaAct, normalized_weights = normalized_weightsAct, data_point_counter_list = data_point_counter_listAct)
                raise Exception("to be checked if 2-task posterior is in this formal")

        # CALCULATING MLLHS
        for modelID in model_set:
            mi = model_set.index(modelID)

            # CALCULATING MLLH FROM THE POSTERIOR FOR THE ACTUAL MODEL
            if modelID == activeModel:
                if activeModel=="x" or activeModel=="y" or activeModel=="1x2D":
                    mllh[mi][t], posteriors[mi], pp = calc_mmllh_1task(dNew, sigma_r, model = modelID, evaluation = 'iterative', posterior = posteriors[model_set.index(winningModel[t-1])], mmllh_prev = mllh[mi][t-1])
                else:
                    raise Exception("for 2-task models the iterative calcuation of posterior is not implemented")
            else:
                # CALCULATING MLLH FROM DREAMED DATA
                for i in range(0,grRepetition):
                    dataGRfused = helper.concatenate_data(dataGR[i], dNew)
                    
                    if modelID == "x" or modelID == "y" or modelID == "1x2D":
                        mllhGR[mi][i], posteriorGR[i] = calc_mmllh_1task(dataGRfused, sigma_r, model = modelID, evaluation = 'full')
                    else:
                        mllhGR[mi][i], posteriorGR[i] = calc_mmllh_2task(dataGRfused, sigma_r, model=modelID, evaluation = "full", num_particles = 256, marginalize = True)
                #print(mllhGR[mi,:])
                #print(np.exp(np.mean(np.log(mllhGR[mi,:]))))
                mllh[mi][t] = np.exp(np.mean(np.log(mllhGR[mi,:])))
                posteriors[mi] = posteriorGR[np.argmax(mllhGR[mi,:])]
        winningModel[t] = model_set[np.argmax(mllh[:,t])]
    return mllh, winningModel
