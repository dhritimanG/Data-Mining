#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:04:27 2018

@author: dhritiman
"""

import numpy as np
import seaborn as sns


def prob_x1_given_x0(mus, sigmas, x0):
    new_mu = mus[1] + sigmas[0][1] * (x0 - mus[0]) / sigmas[0][0]
    new_sigma = sigmas[1][1] - sigmas[0][1] * sigmas[0][1] / sigmas[0][0]
    return np.random.normal(new_mu, new_sigma)
    
def prob_x0_given_x1(mus, sigmas, x1):
    new_mu = mus[0] + sigmas[1][0] * (x1 - mus[1]) / sigmas[1][1]
    new_sigma = sigmas[0][0] - sigmas[1][0] * sigmas[1][0] / sigmas[1][1]
    return np.random.normal(new_mu, new_sigma)
    
def gibbs_sampling(mus, sigmas, prev_x1):
    x0 = prob_x0_given_x1(mus, sigmas, prev_x1)
    x1 = prob_x1_given_x0(mus, sigmas, x0)
    return np.array([[x0, x1]])
    

if __name__ == '__main__':
    mus = np.array([5,5])
    sigmas = np.array([[1,.9],[.8,2]])
    joint = np.random.multivariate_normal(mus, sigmas, 2000)
    fig = sns.jointplot(joint[:,0], joint[:,1], color = 'r')

    # Initialize        
    allsamples = np.zeros([1,2])
    for i in range(100000):
        curr_sample = gibbs_sampling(mus, sigmas, allsamples[-1,1])
        allsamples = np.concatenate((allsamples, curr_sample), axis = 0)
    fig = sns.jointplot(allsamples[:,0], allsamples[:,1], color = 'g')