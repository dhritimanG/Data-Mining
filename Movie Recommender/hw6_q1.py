#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:24:49 2018

@author: dhritiman
"""

import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
col1 = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('/Users/dhritiman/Downloads/ml-100k/u.data', sep = '\t', names = col1)

ratings = ratings.drop('unix_timestamp', axis=1)

ratings.head()
ratings.shape
ratings['user_id'].head()
ratings['user_id'][0]
ratings['movie_id'][0]
ratings['rating'][0]


rating_mat = [[ 0 for j in range(1683)] for i in range(944)]
np.array(rating_mat).shape

rating_mat = np.zeros((944, 1683))

for i in range(len(ratings)):
#    print(i)
    rating_mat[ratings['user_id'][i]][ratings['movie_id'][i]] = ratings['rating'][i]
    
for user_id in range(len(rating_mat)):
    for movie_id in range(len(rating_mat[0])):
        rating_mat[user_id][movie_id] = rating_mat[user_id][movie_id] - np.mean(rating_mat[user_id])


sim_mat = cosine_similarity(rating_mat)

def find_knn(sim_mat, column, k):
    sim_mat_T = sim_mat.T
    # each row in sim_mat_T is a column in sim_mat
    top_k_indices = np.argpartition(sim_mat_T[column], -k)[-k:]   
    return rating_mat[top_k_indices]




## Top k users similar to user 0
#top_similar_raters = find_knn(sim_mat, 0, 10)
#prediction = np.sum(top_similar_raters, axis = 0)/10
#rmse = np.sqrt(mean_squared_error(rating_mat[0], prediction))



# Randomly sample 100 users from user rating matrix:
sample_indices = np.random.randint(0,len(rating_mat), 100)
sample_users = rating_mat[sample_indices]


rmse = 0
rmses = []
k = 10
for i in range(len(sample_users)):
    top_similar_raters = find_knn(sim_mat, i, k)
    prediction = np.sum(top_similar_raters, axis = 0)/k
    rmse = np.sqrt(mean_squared_error(sample_users[i], prediction))
    rmses.append(rmse)


plt.plot(range(1,101), rmses)
