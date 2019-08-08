#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:57:15 2018

@author: dhritiman
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

samples = []
n_samples = 20
population_size = 300
data = np.zeros([population_size, 2])
xmin = 0
xmax = 6

# probability density function
#P = lambda x: np.exp(-x)
P = lambda x: 1/(x+1)

# Generate data
for i in range(population_size):
    #x = np.random.uniform(xmin, xmax)
    data[i][0] = i#x
    data[i][1] = P(i)#P(x)

# Sort them in descending order
#data = data[data[:,1].argsort()[::-1]]
#plt.plot(data[:,0], data[:,1])
temp= np.sum(data[:,1])
data[:,1] = data[:,1]/temp
plt.plot(data[:,0], data[:,1])

# Divide them in different buckets
n_buckets = int(data.shape[0]/n_samples)
data = data.reshape(n_buckets, n_samples, 2)

# Replace the probability distribution in each bucket with uniform probability
data2 = np.zeros([n_buckets, n_samples, 2])
for i in range(n_buckets):
    temp = np.copy(data[i])
    val = np.sum(temp[:,1])
    temp[:,1] = val/n_samples
    data2[i] = np.copy(temp)

# Randomly select buckets with replacement
buckets_picked = []
for i in range(n_samples):
    buckets_picked.append(np.random.randint(0, n_buckets))
    
# Maintain dictionary of buckets that got picked
buckets_picked_dict = dict(Counter(buckets_picked))

# From the buckets that got picked sample records without replacement
for bucket_num in buckets_picked_dict:
    vec = data2[bucket_num][:,0]
    size = buckets_picked_dict[bucket_num]
    samples.extend(np.random.choice(vec, size, replace = False).tolist())
    
samples = np.array(samples)
print(samples)
samples.sort()
