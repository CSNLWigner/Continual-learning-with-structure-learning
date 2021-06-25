import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import zip_longest
import helper

from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def mllh_analytic_1x1D(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), model = 'x'):
    '''
    Computes mllh for 1x1D model.
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
    zs = data['z']
    rs = data['r']
    if zs.size != 0:
        T = zs.shape[0]
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
        return 1. # is this ok?


def mllh_analytic_1x2D(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), return_posterior=False):
    '''
    Analytic computation of marginal likelihood of 1x2D model.
    '''
    zs = data['z']
    rs = data['r']
    if zs.size != 0:
        T = np.size(zs,0)
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
        if return_posterior:
            return {'mllh':y, 'mu_is':mu_is, 'Sigma_i_invs':Sigma_i_invs}
        else:
            return y
    else:
        return 1


def mllh_analytic_2x1D(data, sigma_r, Sigma_0_1D = 1., verbose = True):
    '''
    Analytic computation of marginal likelihood of 2x2D model
    it is validated through 'trial_nonorm_posterior_set_transformed'
    from that function the only step forward is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    '''
    z = data['z']
    r = data['r']
    T = z.shape[0]
  
    indices = np.arange(T)
    index_subsets = list(powerset(indices))

    mmllh_accumulator = 0.
    if verbose:
        pbar = tf.keras.utils.Progbar(len(index_subsets))
    for index_subset in index_subsets:
        zx = z[list(index_subset)]
        rx = r[list(index_subset)]
        datax = {'z':zx, 'r':rx}

        complementer_subset = [item for item in indices if item not in index_subset]

        zy = z[complementer_subset]
        ry = r[complementer_subset]
        datay = {'z':zy, 'r':ry}

        mmllh_accumulator += mllh_analytic_1x1D(datax, sigma_r, Sigma_0 = Sigma_0_1D, model = 'x') \
            * mllh_analytic_1x1D(datay, sigma_r, Sigma_0 = Sigma_0_1D, model = 'y')
    
        if verbose:
            pbar.add(1)
    mmllh_accumulator /= 2**T
    return mmllh_accumulator


def mllh_analytic_2x2D(data, sigma_r, Sigma_0_2D = np.array([[1., 0.], [0., 1.]]), verbose = True):
    '''
    Analytic computation of marginal likelihood of 2x2D model
    it is validated through 'trial_nonorm_posterior_set_transformed'
    from that function the only step forward is to leave the normal in gamma (the gamma posterior) since gamma is marginalized out
    '''
    z = data['z']
    r = data['r']
    T = z.shape[0]
    
    indices = np.arange(T)
    index_subsets = list(powerset(indices))

    mmllh_accumulator = 0.
    if verbose:
        pbar = tf.keras.utils.Progbar(len(index_subsets))
    for index_subset in index_subsets:
        z1 = z[list(index_subset)]
        r1 = r[list(index_subset)]
        data1 = {'z':z1, 'r':r1}
        
        complementer_subset = [item for item in indices if item not in index_subset]
        
        z2 = z[complementer_subset]
        r2 = r[complementer_subset]
        data2 = {'z':z2, 'r':r2}
        
        mmllh_accumulator += mllh_analytic_1x2D(data1, sigma_r, Sigma_0 = Sigma_0_2D) \
        * mllh_analytic_1x2D(data2, sigma_r, Sigma_0 = Sigma_0_2D)
        
        if verbose:
            pbar.add(1)
            
    mmllh_accumulator /= 2**T
    return mmllh_accumulator


def mllh_analytic_2x2D_shared(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]]), return_posterior=False):
    '''
    Computes mllh for 2x2D-shared decision boundary model.
    https://www.notion.so/2x2D-shared-Shared-decision-boundary-model-888ee041f7954cf0a2b98fa7db774ebc
    '''
    zs = data['z']
    rs = data['r']
    bs = data['b']
    if zs.size != 0:
        T = np.size(zs,0)
        assert not np.isscalar(Sigma_0), 'Sigma_0 must be a 2-dimensional array'
        detSigma_0 = np.linalg.det(Sigma_0)
        Sigma_i_star_invs = []
        Sigma_i_invs = []
        mu_is = []
        y = 1/(2*np.pi)/np.sqrt(np.linalg.det(Sigma_0))
        for t in range(T):
            z = zs[t]
            r = rs[t]
            b = bs[t]
            Sigma_i_star_inv = np.array([[z[0]**2/sigma_r**2, z[0]*z[1]/sigma_r**2],[z[0]*z[1]/sigma_r**2, z[1]**2/sigma_r**2]])
            Sigma_i_star_invs.append(Sigma_i_star_inv)
            if t==0:
                Sigma_i_inv = Sigma_i_star_inv + np.linalg.inv(Sigma_0)
            else:
                Sigma_i_inv = Sigma_i_star_inv + Sigma_i_invs[t-1]
            Sigma_i_invs.append(Sigma_i_inv)
            Sigma_i = np.linalg.inv(Sigma_i_inv)
            if t==0:
                mu_i = Sigma_i.dot(pow(-1,b)*z*r/sigma_r**2)
            else:
                mu_i = Sigma_i.dot(pow(-1,b)*z*r/sigma_r**2 + Sigma_i_invs[t-1].dot(mu_is[t-1]))
            mu_is.append(mu_i)
            y = y * multivariate_normal.pdf(r, mean = 0, cov = sigma_r**2)
        y = y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i)
        if return_posterior:
            return {'mllh':y, 'mu_is':mu_is, 'Sigma_i_invs':Sigma_i_invs}
        else:
            return y
    else:
        return 1


def mllh_analytic_2x1D_observed_context(data, sigma_r, Sigma_0_1D = 1., verbose = True):
    '''
    Analytic computation of marginal likelihood of 2x1D model where it is assumed that background corresponds directly to context.
    '''
    z = data['z']
    r = data['r']
    b = data['b']
    T = z.shape[0]

    datax = helper.reorder_data(data,np.where(data['b'] == 0)[0]) #check if 0 corresponds to the right task!
    datay = helper.reorder_data(data,np.where(data['b'] == 1)[0])

    mmllh_accumulator = mllh_analytic_1x1D(datax, sigma_r, Sigma_0 = Sigma_0_1D, model = 'x') \
        * mllh_analytic_1x1D(datay, sigma_r, Sigma_0 = Sigma_0_1D, model = 'y')
        
    return mmllh_accumulator


def mllh_analytic_2x2D_observed_context(data, sigma_r, Sigma_0_1D = 1., verbose = True):
    '''
    Analytic computation of marginal likelihood of 2x2D model where it is assumed that background corresponds directly to context.
    '''
    z = data['z']
    r = data['r']
    b = data['b']
    T = z.shape[0]

    datax = helper.reorder_data(data,np.where(data['b'] == 0)[0]) #check if 0 corresponds to the right task!
    datay = helper.reorder_data(data,np.where(data['b'] == 1)[0])

    mmllh_accumulator = mllh_analytic_1x2D(datax, sigma_r, Sigma_0 = Sigma_0_1D) \
        * mllh_analytic_1x2D(datay, sigma_r, Sigma_0 = Sigma_0_1D)
        
    return mmllh_accumulator