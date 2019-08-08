# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time


file1 = open("/Users/dhritiman/Documents/Jupyter Notebook files/kosarak.dat", "r")
#file1 = open(sys.argv[1],"r")

lines = file1.read().splitlines()

l = []
max_num = 0
tmp = []

time_start = time.time()
for line in lines:
    tmp = list(set(map(int, line.split())))
    tmp.sort()
    l.append(tmp)
    for num in tmp:
        if num > max_num:
            max_num = num
    
outputFile = open("/Users/dhritiman/Documents/Jupyter Notebook files/kosarak_output.arff", "w")

    
outputList = ['@RELATION test']
for i in range(1,max_num+1):
    outputList.append('@ATTRIBUTE i%d {0, 1}' %i)
    
outputList.append('@DATA')


for i in outputList:
    outputFile.writelines(i+'\n')

for i in l:
    tmp = '{'
    for j in range(len(i)):
        tmp += str(i[j]-1) + " 1, "
    tmp = tmp[:-2]
    tmp += "}\n"
    outputFile.writelines(tmp)

time_end = time.time()
time_delta = time_end - time_start
    
outputFile.close()