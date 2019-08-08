#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:42:41 2018

@author: dhritiman
"""
import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from __future__ import division

# NG TRAIN
newsgroups_train = fetch_20newsgroups(subset='train')

from pprint import pprint
pprint(list(newsgroups_train.target_names))

newsgroups_train.filenames.shape
newsgroups_train.target.shape

from sklearn.feature_extraction.text import TfidfVectorizer
#categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train')
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors.shape

# NG TEST
newsgroups_test = fetch_20newsgroups(subset='test')
vectors_test = vectorizer.transform(newsgroups_test.data)
vectors_test.shape

# Vectorize
vectorizer = TfidfVectorizer(use_idf = False).fit(newsgroups_train.data)
train_NG = vectorizer.transform(newsgroups_train.data)
test_NG = vectorizer.transform(newsgroups_test.data)
train_NG.shape
test_NG.shape


# NF LABELS
train_labels = newsgroups_train.target
test_labels = newsgroups_test.target
train_labels.shape
test_labels.shape

#Dot Product
def compute_dot_product(traininig_mat, test_mat):
    time_start = time.time()
    dot_result = np.dot(traininig_mat, test_mat.T)
    time_end = time.time()
    time_delta = time_end - time_start
    print(time_delta)
    return dot_result

#Cosine similarity:
#def compute_cosine_sim(traininig_mat, test_mat):
#    time_start = time.time()
#    x_train_norm = np.linalg.norm(traininig_mat, axis = 1, keepdims = True)
#    x_test_norm = np.linalg.norm(test_mat, axis = 1, keepdims = True)
#    dot_result_cosine = np.dot(traininig_mat, test_mat.T)
#    dot_result_cosine_norm = np.dot(x_train_norm, x_test_norm.T)
#    cosine_dist = dot_result_cosine/dot_result_cosine_norm
#    time_end = time.time()
#    time_delta = time_end - time_start
#    print(time_delta)
#    return cosine_dist

dot_result = compute_dot_product(vectors,vectors_test)
cosine_result = cosine_similarity(vectors,vectors_test)

dot_result[0].shape

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

type(dot_result)
dot_result = dot_result.todense()
type(dot_result)
dot_result = np.array(dot_result)
type(dot_result)
dot_result.shape
knn_result = find_knn(dot_result, 0, 10)

cosine_result.shape

#Accuracy DOT
good = 0
for i in range(0, len(dot_result[0])):
    if test_labels[i] == find_knn(dot_result, i, 10):
        good += 1
print good
accuracy = good/len(dot_result[0])
print(accuracy)

#Accuracy COSINE
good = 0
for i in range(0, len(dot_result[0])):
    if test_labels[i] == find_knn(cosine_result, i, 10):
        good += 1
print good
accuracy = good/len(dot_result[0])
print(accuracy)

