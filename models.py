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


def mllh_analytic_1x1D(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]])):
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
    return 1


def mllh_analytic_1x2D(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]])):
    if zs.size != 0:
        T = np.size(zs,0)
        assert not np.isscalar(Sigma_0), 'Sigma_0 must be a 2-dimensional array'
        detSigma_0 = np.linalg.det(Sigma_0)
        Sigma_i_star_invs = []
        Sigma_i_invs = []
        mu_is = []
        y = 1/(2*np.pi)/np.sqrt(np.linalg.det(Sigma_0))
        zs = data['z']
        rs = data['r']
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
        return y
    else:
        return 1


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
        
        complementer_subset = [item for item in indices if item not in index_subset]
        
        z2 = z[complementer_subset]
        r2 = r[complementer_subset]
        
        mmllh_accumulator += mllh_analytic_1x2D(z1, r1, sigma_r, Sigma_0 = Sigma_0_2D) \
        * mllh_analytic_1x2D(z2, r2, sigma_r, Sigma_0 = Sigma_0_2D)
        
        if verbose:
            pbar.add(1)
            
    mmllh_accumulator /= 2**T
    return mmllh_accumulator

def mllh_analytic_2x2D_shared(data, sigma_r, Sigma_0 = np.array([[1., 0.], [0., 1.]])):
    '''
    Computes mllh for 2x2D-shared decision boundary model.
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
                mu_i = Sigma_i.dot(pow(-1,0)*z*r/sigma_r**2)
            else:
                mu_i = Sigma_i.dot(pow(-1,0)*z*r/sigma_r**2 + Sigma_i_invs[t-1].dot(mu_is[t-1]))
            mu_is.append(mu_i)
            y = y * multivariate_normal.pdf(r, mean = 0, cov = sigma_r**2)
        y = y / multivariate_normal.pdf(mu_i, mean = np.array([0,0]), cov = Sigma_i)
        return y
    else:
        return 1
