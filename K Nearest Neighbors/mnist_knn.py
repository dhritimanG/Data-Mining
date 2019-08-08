#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:56:32 2018

@author: dhritiman
"""

import scipy.misc
import numpy as np
import time
from __future__ import division

images_train = train_images()
images_test = test_images()

train_labels = train_labels()
test_labels = test_labels()

x_train = images_train.reshape((images_train.shape[0], images_train.shape[1] * images_train.shape[2]))
x_test = images_test.reshape((images_test.shape[0], images_test.shape[1] * images_test.shape[2]))
x_test_T = x_test.T

#x_train.shape
#x_test.shape


## Dot product similarity
def compute_dot_product(traininig_mat, test_mat):
    time_start = time.time()
    dot_result = np.dot(traininig_mat, test_mat.T)
    time_end = time.time()
    time_delta = time_end - time_start
    print(time_delta)
    return dot_result
    
dot_result = compute_dot_product(x_train, x_test)
dot_result.shape
type(dot_result)

## Cosine similarity:
def compute_cosine_sim(traininig_mat, test_mat):
    time_start = time.time()
    x_train_norm = np.linalg.norm(traininig_mat, axis = 1, keepdims = True)
    x_test_norm = np.linalg.norm(test_mat, axis = 1, keepdims = True)
    dot_result_cosine = np.dot(traininig_mat, test_mat.T)
    dot_result_cosine_norm = np.dot(x_train_norm, x_test_norm.T)
    cosine_dist = dot_result_cosine/dot_result_cosine_norm
    time_end = time.time()
    time_delta = time_end - time_start
    print(time_delta)
    return cosine_dist
    
cosine_result = compute_cosine_sim(x_train, x_test)


# Find knn:
def find_knn(sim_mat, column, k):
    sim_mat_T = sim_mat.T
    # each row in sim_mat_T is a column in sim_mat
    top_k_indices = np.argpartition(sim_mat_T[column], -k)[-k:]
#    print("labels of training set corresponding to top k indices:\n")
#    print(train_labels[top_k_indices])
#    print("\n")
#    print("label in test set:\n")
#    print(test_labels[column])
    counts = np.bincount(train_labels[top_k_indices])
    return np.argmax(counts)
    
knn_result = find_knn(dot_result, 0, 10)


# Accuracy DOT:
good = 0
for i in range(0, len(dot_result[0])):
    if test_labels[i] == find_knn(dot_result, i, 10):
        good += 1
print good
accuracy = good/len(dot_result[0])
print(accuracy)

# Accuracy COSINE:
good = 0
for i in range(0, len(dot_result[0])):
    if test_labels[i] == find_knn(cosine_result, i, 10):
        good += 1
print good
accuracy = good/len(dot_result[0])
print(accuracy)

    