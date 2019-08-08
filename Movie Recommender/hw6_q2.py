#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:38:37 2018

@author: dhritiman
"""

#import igraph
import networkx as nx
import pandas as pd
import json

#from igraph import *
#from networkx import *


graph_edges = pd.read_csv('/Users/dhritiman/Downloads/Flickr_sampled_edges/edges_sampled_2K.csv', sep=',', header=None)

graph_edges.columns.values
len(graph_edges.index)

vertices = set()
for i in range(len(graph_edges)):
    vertices.add(graph_edges[0][i])
    vertices.add(graph_edges[1][i])


edges1 = [(graph_edges[0][i], graph_edges[1][i]) for i in range(len(graph_edges))]


#g = Graph(vertex_attrs={"label":vertices}, edges=edges1, directed=True)
#communities = g.community_edge_betweenness(directed=True)

g_nx = nx.Graph()
g_nx.add_nodes_from(vertices)
g_nx.add_edges_from(edges1)

# Test for shortest paths:
#print([p for p in nx.all_shortest_paths(g_nx, source=27228, target=33754)])


#nodes = g_nx.nodes()
#len(nodes)
#
#shortest_paths = dict()
#for i in range(0, len(nodes)):
#    for j in range(i+1, len(nodes)):
#        shortest_paths[(nodes[i],nodes[j])] = nx.all_shortest_paths(g_nx, source=nodes[i], target=nodes[j])


## Version 5
#
#shortest_paths = dict()
#for vertex1 in vertices:
#    all_edges_len = len(edges1)
#    v1_edge_set = []
#    target = []
#    for i in range(all_edges_len):
#        if vertex1 == edges1[i][0]:
#            v1_edge_set.append(edges1[i])
#            target.append(edges1[i][1])
#    for vertex2 in target:
#        print((vertex1,vertex2))
#        if(nx.has_path(g_nx, vertex1,vertex2)):
#            print((vertex1, vertex2))
#            L = [p for p in nx.all_shortest_paths(g_nx, source=vertex1, target=vertex2)]
#            tuples = []
#            for item in L:
#                tuples.append([(x, y) for x in item for y in item if item.index(y) == item.index(x)+1])
#            shortest_paths[(vertex1,vertex2)] = tuples
##    break
#    print(vertex1)



# Version 4

shortest_paths = dict()
node1 = set(graph_edges[0])
for vertex1 in node1:
    for vertex2 in graph_edges[1]:
#        print((vertex1,vertex2))
        if(nx.has_path(g_nx, vertex1,vertex2)):
            print((vertex1, vertex2))
            L = [p for p in nx.all_shortest_paths(g_nx, source=vertex1, target=vertex2)]
            tuples = []
            for item in L:
                tuples.append([(x, y) for x in item for y in item if item.index(y) == item.index(x)+1])
            shortest_paths[(vertex1,vertex2)] = tuples
#    break
    print(vertex1)


# Version 3
#
#shortest_paths = dict()
#node1 = set(graph_edges[0])
#for vertex1 in node1:
#    all_edges_len = len(edges1)
#    v1_edge_set = []
#    target = []
#    for i in range(all_edges_len):
#        if vertex1 == edges1[i][0]:
#            v1_edge_set.append(edges1[i])
#            target.append(edges1[i][1])
#    for vertex2 in target:
#        print((vertex1,vertex2))
#        if(nx.has_path(g_nx, vertex1,vertex2)):
#            L = [p for p in nx.all_shortest_paths(g_nx, source=vertex1, target=vertex2)]
#            tuples = []
#            for item in L:
#                tuples.append([(x, y) for x in item for y in item if item.index(y) == item.index(x)+1])
#            shortest_paths[(vertex1,vertex2)] = tuples
##        break
##    break
#    print(vertex1)



# Version 2

#shortest_paths = dict()
#for vertex1 in graph_edges[0]:
#    all_edges_len = len(edges1)
#    v1_edge_set = []
#    target = []
#    for i in range(all_edges_len):
#        if vertex1 == edges1[i][0]:
#            v1_edge_set.append(edges1[i])
#            target.append(edges1[i][1])
#    for vertex2 in target:
#        print((vertex1,vertex2))
#        if(nx.has_path(g_nx, vertex1,vertex2)):
#            L = [p for p in nx.all_shortest_paths(g_nx, source=vertex1, target=vertex2)]
#            tuples = []
#            for item in L:
#                tuples.append([(x, y) for x in item for y in item if item.index(y) == item.index(x)+1])
#            shortest_paths[(vertex1,vertex2)] = tuples
##        break
##    break
#    print(vertex1)


# Version 1

#shortest_paths = dict()
#for vertex1 in graph_edges[0]:
#    for vertex2 in graph_edges[1]:
##        print((vertex1,vertex2))
#        if(nx.has_path(g_nx, vertex1,vertex2)):
#            print((vertex1, vertex2))
#            L = [p for p in nx.all_shortest_paths(g_nx, source=vertex1, target=vertex2)]
#            tuples = []
#            for item in L:
#                tuples.append([(x, y) for x in item for y in item if item.index(y) == item.index(x)+1])
#            shortest_paths[(vertex1,vertex2)] = tuples
##    break
#    print(vertex1)
#
#
#
shortest_paths[(27228,33754)]
shortest_paths[(27228,27340)]

#shortest_paths_to_file = {str(shortest_paths.keys()) : shortest_paths.values()}
#with open('shortest_path_dict.txt', 'w') as file:
#     file.write(json.dumps(shortest_paths_to_file))


# Define a dictionary for ratio of all shortest paths passign through edge e:
betweennness_e = dict()
fraction = 0
for edge in edges1:
    numerator = 0
    denominator = 0
    for vertex1 in node1:
        for vertex2 in graph_edges[1]:
            if(nx.has_path(g_nx, vertex1,vertex2)):
                print(vertex1)
                denominator = len(shortest_paths[vertex1,vertex2])
                if edge in shortest_paths[vertex1,vertex2]:
                    numerator = numerator+1
                fraction += numerator/denominator
    betweennness_e[edge] = fraction
            
                





# Tests
#g_nx_test = nx.Graph()
#g_nx_test.add_path([0,1,2])
#g_nx_test.add_path([0,5,2])
#g_nx_test.add_path([0,6,4,2])
#g_nx_test.add_path([0,8,9,2])
##g_nx_test.add_path([0,2])
#print([p for p in nx.all_shortest_paths(g_nx_test, source=0, target=2)])