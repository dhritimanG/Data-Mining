#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:13:36 2018

@author: dhritiman
"""

file = open("/Users/dhritiman/Documents/Jupyter Notebook files/AP_train.txt", "r")
lines = file.readlines()

authors = []
pub_venue = []
pubs = []
#auth_pub_dict = {}
pubs_for_auth = {}

lineNo = 0
for line in lines:
    line = line.strip()
    
    if("#index" in line):
       line = line.split(";")
       line[0] = line[0].split("#index")
       line[0] = ''.join(line[0])
       line = ','.join(line)
       pubs.append(line)
       
    if("#c" in line):
       line = line.split(";")
       line[0] = line[0].split("#c")
       line[0] = ''.join(line[0])
       line = ','.join(line)
       pub_venue.append(line)
       
    if("#@" in line):
       line = line.split(";")
       line[0] = line[0].split("#@")
       line[0] = ''.join(line[0])
       line = ','.join(line)
       authors.append(line)
#       pubs_for_auth.append(pubs[lineNo])
#       auth_pub_dict[authors[lineNo]] = pubs_for_auth
#       lineNo += 1

authors = set(authors)
pub_venue = set(pub_venue)
pubs = set(pubs)

# Length of each:
len_authors = len(set(authors))
len_pub_venue = len(set(pub_venue))
len_publications = len(set(pubs))


    
