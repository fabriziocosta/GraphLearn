import networkx as nx
from eden.graph import Vectorizer
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
import random
#from eden.model import ActiveLearningBinaryClassificationModel
from numpy.random import randint as p_randint
from numpy.random import uniform as p_uniform
import sklearn.metrics as metrics
'''
    wrapped or altered eden functions:
        -my_vectorizer, a eden vectorizer that doesnt try to expand graphs
        -my_fit_estimator, a estimator that doesnt do validation and doesnt print 
        -expand_edges, just for convenience.
        
'''


class GraphLearnVectorizer(Vectorizer):
    '''
    doing some overwriting so we dont expand and contract edges all the time..
    this hack is a little bit dependant on the state of eden.. so be carefull here 
    '''

    def _edge_to_vertex_transform(self, graph):
        return nx.Graph(graph)

    def transform2(self, graph):
        return self._convert_dict_to_sparse_matrix(self._transform(0, graph))




def expand_edges(graph):
    '''
    convenience wrapper
    '''
    vectorizer = Vectorizer(complexity= 3)
    return vectorizer._edge_to_vertex_transform(graph)


def contract_edges(original_graph):
    """
        stealing from eden...
        because i draw cores and interfaces there may be edge-nodes 
        that have no partner, eden gives error in this case.
        i still want to see them :) 
    """
    # start from a copy of the original graph
    G = nx.Graph(original_graph)
    # re-wire the endpoints of edge-vertices
    for n, d in original_graph.nodes_iter(data=True):
        if d.get('edge', False) == True:
            # extract the endpoints
            endpoints = [u for u in original_graph.neighbors(n)]
            # assert (len(endpoints) == 2), 'ERROR: more than 2 endpoints'
            if len(endpoints) != 2:
                continue
            u = endpoints[0]
            v = endpoints[1]
            # add the corresponding edge
            G.add_edge(u, v, d)
            # remove the edge-vertex
            G.remove_node(n)
        if d.get('node', False) == True:
            # remove stale information
            G.node[n].pop('remote_neighbours', None)
    return G



def select_random(graph_iter, len,samplesize):
    x=range(len)
    random.shuffle(x)
    x=x[:samplesize]
    x.sort(reverse=True)
    next=x.pop()
    for i,g in enumerate(graph_iter):
        if i==next:
            yield g
            if not x:
                break
            next=x.pop()

from collections import defaultdict

def calc_stats_from_grammar(grammar):
    count_corehashes = defaultdict(int)
    count_interfacehashes = defaultdict(int)
    corecounter = defaultdict(int)
    intercounter = defaultdict(int)
    for ih in grammar.keys():
        for ch in grammar[ih].keys():
            # go over all the combos
            count_corehashes[ch]+=1
            count_interfacehashes[ih]+=1
            count= grammar[ih][ch].count
            corecounter[ch]+=count
            intercounter[ih]+=count
    return count_corehashes,count_interfacehashes,corecounter,intercounter






