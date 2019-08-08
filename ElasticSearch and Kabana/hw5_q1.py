#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 23:03:38 2018

@author: dhritiman
"""

import glob
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation   
from collections import Counter
import math
from string import punctuation
import numpy as np


dirpath = "/Users/dhritiman/Downloads/DUC2001/Summaries/"

#ps = PorterStemmer()
es = Elasticsearch()
ip = glob.glob(dirpath + "*")
doc_ids = []
for i in ip:
    if i != '/Users/dhritiman/Downloads/DUC2001/Summaries/ap900322-0200_system.txt':
        file = open(i, 'r')
        soup = BeautifulSoup(file, 'html.parser')
        corpus = soup.getText().split('Introduction:\n')
        doc_id = i.split("Summaries/")[1].split(".txt")[0].upper()
        gold_summary = corpus[0].split('Abstract:\n')[1].lstrip(' ')
        print(i)
        doc_text = corpus[1]
        if len(doc_text.split()) > 5:
            es_doc = {
                      'doc_id': doc_id,
                      'gold_summary': gold_summary,
                      'doc_text': doc_text
                      }
            res = es.index(index = "duc_dataset",
                           doc_type = "document",
                           id = doc_id,
                           body = es_doc)


from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='all')
len(newsgroups_train.data)
newsgroups_train.target_names
len(newsgroups_train.filenames)

import ntpath
all_files = []
all_files.append(ntpath.basename(newsgroups_train.filenames[0]))


for i in range(len(newsgroups_train.data)):
    doc_id = ntpath.basename(newsgroups_train.filenames[i])
    doc_text = newsgroups_train.data[i].strip("\n")
    es_doc = {
              'doc_id': doc_id,
              'doc_text': doc_text
              }
    
    res = es.index(index = "20ng_dataset",
                           doc_type = "document",
                           id = doc_id,
                           body = es_doc)