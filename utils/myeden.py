import networkx as nx
from eden.graph import Vectorizer
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
import random
'''
    wrapped or altered eden functions:
        -my_vectorizer, a eden vectorizer that doesnt try to expand graphs
        -my_fit_estimator, a estimator that doesnt do validation and doesnt print 
        -expand_edges, just for convenience.
        
'''


class my_vectorizer(Vectorizer):
    '''
    doing some overwriting so we dont expand and contract edges all the time..
    this hack is a little bit dependant on the state of eden.. so be carefull here 
    '''

    def _edge_to_vertex_transform(self, graph):
        return nx.Graph(graph)

    def transform2(self, graph):
        return self._convert_dict_to_sparse_matrix(self._transform(0, graph))



def my_fit_estimator(positive_data_matrix=None, negative_data_matrix=None, target=None, cv=10, n_jobs=-1):
    '''
     we dont need the validation at all... so i just copied from eden/utils/__init__.py
    '''
    assert (
        positive_data_matrix is not None), 'ERROR: expecting non null positive_data_matrix'
    if target is None and negative_data_matrix is not None:
        yp = [1] * positive_data_matrix.shape[0]
        yn = [-1] * negative_data_matrix.shape[0]
        y = np.array(yp + yn)
        X = vstack([positive_data_matrix, negative_data_matrix], format="csr")
    if target is not None:
        X = positive_data_matrix
        y = target

    predictor = SGDClassifier(class_weight='auto', shuffle=True, n_jobs=n_jobs)
    # hyperparameter optimization
    param_dist = {"n_iter": randint(5, 100),
                  "power_t": uniform(0.1),
                  "alpha": uniform(1e-08, 1e-03),
                  "eta0": uniform(1e-03, 10),
                  "loss":[ 'log'],
                  "penalty": ["l1", "l2", "elasticnet"],
                  "learning_rate": ["invscaling", "constant", "optimal"]}
    scoring = 'roc_auc'
    n_iter_search = 20
    random_search = RandomizedSearchCV(
        predictor, param_distributions=param_dist, n_iter=n_iter_search, cv=cv, scoring=scoring, n_jobs=n_jobs)
    random_search.fit(X, y)
    optpredictor = SGDClassifier(
        class_weight='auto', shuffle=True, n_jobs=n_jobs, **random_search.best_params_)
    # fit the predictor on all available data
    optpredictor.fit(X, y)
    return optpredictor


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



