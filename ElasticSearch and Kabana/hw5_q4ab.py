#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 06:00:57 2018

@author: dhritiman
"""

import numpy as np
import matplotlib.pyplot as plt

def uniform_distribution_sampling(xmin, xmax, sample_size):
    P = lambda x: 1/(xmax-xmin)
    
    # domain limits
    xmin = 2 # the lower limit of our domain
    xmax = 5 # the upper limit of our domain
    
    # range limit (supremum) for y
    ymax = 1
    #you might have to do an optimization to find this.
    
    N = 10000 # the total of samples we wish to generate
    accepted = 0 # the number of accepted samples
    samples = np.zeros(N)
    count = 0 # the total count of proposals
    
    # generation loop
    while (accepted < N):
        
        # pick a uniform number on [xmin, xmax) (e.g. 0...10)
        x = np.random.uniform(xmin, xmax)
        
        # pick a uniform number on [0, ymax)
        y = np.random.uniform(0,ymax)
        
        # Do the accept/reject comparison
        if y < P(x):
            samples[accepted] = x
            accepted += 1
        
        count +=1
        
    print("Count",count, "Accepted", accepted)
    
    # get the histogram info
    #hinfo = np.histogram(samples,30)
    
    # plot the histogram
    plt.hist(samples,bins=30, label=u'Samples');


xmin = 2
xmax = 5
sample_size = 10000
uniform_distribution_sampling(xmin, xmax, sample_size)
######################################################################

def gaussian_distribution_sampling(mu, sigma, sample_size):
    # probability density function
    P = lambda x, mu, sigma: np.exp(-np.square(x-mu)/(2*np.square(sigma)))/(np.sqrt(2*np.pi) * sigma)
    
    # range limit (supremum) for y
    ymax = 1
    
    accepted = 0 # the number of accepted samples
    samples = np.zeros(sample_size)
    z = np.zeros(sample_size)
    count = 0 # the total count of proposals
    
    xmin = mu - (sigma * 5)
    xmax = mu + (sigma * 5)
    
    # generation loop
    while (accepted < sample_size):
        
        # pick a uniform number on [xmin, xmax) (e.g. 0...10)
        x = np.random.uniform(xmin, xmax)
        
        # pick a uniform number on [0, ymax)
        y = np.random.uniform(0, ymax)
        
        # Do the accept/reject comparison
        p = P(x, mu, sigma)
        if y < p:
            samples[accepted] = x
            z[accepted] = p
            accepted += 1
        
        count +=1
        
    print("Count",count, "Accepted", accepted)
    
    # get the histogram info
    #hinfo = np.histogram(samples, 30)
    
    # plot the histogram
    plt.hist(samples, bins=30, label=u'Samples')
    plt.show()
    
    # plot our (normalized) function
    #xvals=np.linspace(xmin, xmax, 1000)
    #plt.plot(xvals, hinfo[0][0]*P(xvals, mu, sigma), 'r', label=u'P(x)')
    #plt.show()

    #return samples

mu = 3
sigma = 1
sample_size = 10000
gaussian_distribution_sampling(3, 1, sample_size)

######################################################################    

def gaussian_2dim_distribution_sampling(mu, cov, sample_size):
    # probability density function
    P = lambda x, mu, cov: np.exp(-0.5 * np.dot(np.dot((x-mu), np.linalg.pinv(cov)), (x-mu).T))/(np.sqrt(np.abs(np.linalg.det(cov))) * 2 * np.pi)
    
    # range limit (supremum) for y
    ymax = 3
    
    accepted = 0 # the number of accepted samples
    samples = np.zeros(shape = (sample_size,2))
    count = 0 # the total count of proposals
    z = np.zeros(sample_size)
    
    xmin = min(mu) - (cov.max() * 5)
    xmax = max(mu) + (cov.max() * 5)
    
    # generation loop
    while (accepted < sample_size):
        
        # pick a uniform number on [xmin, xmax) (e.g. 0...10)
        x = np.random.uniform(xmin, xmax, 2)
        
        # pick a uniform number on [0, ymax)
        y = np.random.uniform(0, ymax)
        
        # Do the accept/reject comparison
        p = P(x, mu, cov)
        if y < p:
            samples[accepted] = x
            z[accepted] = p
            accepted += 1
            #if accepted % 500 == 0:
#            print(accepted)
        
        count +=1
        
    print("Count",count, "Accepted", accepted)
    
    # get the histogram info
    #hinfo = np.histogram(samples, 30)
    
    # plot the histogram
    plt.hist(samples, bins=30, label=u'Samples')
    plt.show()
    
    # Plot scatter plot
    plt.scatter(samples[:,0], samples[:,1], marker = u'x', color = 'r')
    plt.show()
    # plot our (normalized) function    
    #xvals=np.linspace(xmin, xmax, 999)
    #plt.plot(xvals, hinfo[0][0]*P(xvals, mu, sigma), 'r', label=u'P(x)')

    #return samples
    
mu = np.array([3, 3])
cov = np.array([[1, 0],[0, 3]])
gaussian_2dim_distribution_sampling(mu, cov, 1000)
