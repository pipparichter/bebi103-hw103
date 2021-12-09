import numpy as np
import pandas as pd
import scipy 
import scipy.stats
import numpy.random
import os
import warnings
from scipy.special import logsumexp

# Initialize a random number generator.                                 
rg = np.random.default_rng()

def log_llh_gamma(params, data): 
    '''
    Log-likelihood function for a Gamma distribution. 
    
    Params
    ------
    params : tuple
        A two-tuple containing the alpha and beta values, respectively
        for the Gamma distribution. 
    data : np.array
        A n-by-2 np.array containing the data necessary to compute the 
        log-likelihood. 
    '''
    a,b = params
    a,b = np.exp(a), np.exp(b)
    
    return np.sum(scipy.stats.gamma.logpdf(data, a, loc=0, scale=1/b))

def mle(data, log_llh, x0):
    '''
    
    '''
    fun = lambda params : -log_llh(params, data) 

    res = scipy.optimize.minimize(
        fun=fun, 
        x0=x0,
        method = "Powell"
    ) 
    return np.exp(res.x)

def run_bootstrap(data, f, runs=1000):
    res = []
    for run in range(runs):
        rep = np.random.choice(data, len(data), replace=True)
        res.append(f(rep))
    return res

def gamma_sample(params, size=100): 
    '''
    Gets a random sample of a specified size from a Gamma distrinution
    with the given parameters 
    '''
    a,b = params
    sample = np.zeros(size)
    for i in range(size): 
        sample[i] = rg.gamma(a, scale=1/b)
    return sample

def log_llh_joint_poisson(params, data):
    '''
    '''
    B1, B2 = np.exp(params)
    if abs(B2 - B1) <0.000005  or B1<=0 or B2 <=0:
        return -np.inf
    
    return np.sum(np.log((B1*B2/(B2-B1))*(np.exp(-B1*data) - np.exp(-B2*data))))

def get_successive_poisson_params(data): 
    '''
    Get the parameters for the joint Poisson distribution used to model
    bacterial growth. 
    
    Params
    ------
    data : np.array
        n by 2 array used which stores the data to fit. 
    '''    
    res = mle(data, log_llh_joint_poisson, np.array([-5,-4]))
    B1, B2 = res[0], res[1]
    return (B1, B2)

def get_bootstrapped_gamma_params(data, upper=97.5, lower=2.5):
    '''
    Obtains the confidence intervals for the data for a single
    concentration. Also computes the mean values, and returns those. 
    '''
    res = {}
    a_arr, b_arr = [], []
    for run in range(10):
        rep = np.random.choice(data, len(data), replace=True)
        a, b = mle(rep, log_llh_gamma, np.ones(2))
        a_arr.append(a)
        b_arr.append(b)
    return (a_arr, b_arr)
    

    # conf_int_a = np.percentile(a_arr, [lower, upper])
    # conf_int_b = np.percentile(b_arr, [lower, upper])
    
    # res['a'] = {'mean':np.mean(a_arr), 'conf_int':conf_int_a}
    # res['b'] = {'mean':np.mean(b_arr), 'conf_int':conf_int_b}
    
    # return res