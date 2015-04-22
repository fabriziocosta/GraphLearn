import networkx as nx
from eden.graph import Vectorizer
import numpy as np
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
import random
from eden.model import ActiveLearningBinaryClassificationModel
from numpy.random import randint as p_randint
from numpy.random import uniform as p_uniform
import sklearn.metrics as metrics
'''
    wrapped or altered eden functions:
        -my_vectorizer, a eden vectorizer that doesnt try to expand graphs
        -my_fit_estimator, a estimator that doesnt do validation and doesnt print 
        -expand_edges, just for convenience.
        
'''


class MyVectorizer(Vectorizer):
    '''
    doing some overwriting so we dont expand and contract edges all the time..
    this hack is a little bit dependant on the state of eden.. so be carefull here 
    '''

    def _edge_to_vertex_transform(self, graph):
        return nx.Graph(graph)

    def transform2(self, graph):
        return self._convert_dict_to_sparse_matrix(self._transform(0, graph))


''' MAYBE we use this in the future
def my_fit_estimator_model(positive_data_matrix=None, negative_data_matrix=None, target=None, cv=10, n_jobs=-1):


    vectorizer=MyVectorizer()

    estimator = SGDClassifier(average=True, class_weight='auto', shuffle=True)
    pre_processor= lambda x:x
    model = ActiveLearningBinaryClassificationModel(pre_processor=pre_processor,
                                                    estimator=estimator,
                                                    vectorizer=vectorizer,
                                                    n_jobs=n_jobs,

                                                    n_blocks=5)

    #optimize hyperparameters and fit model

    pre_processor_parameters={}
                            #'max_num':[1,3],
                            #  'shape_type':[5],
                            #  'energy_range':[5,10,20,30]}

    vectorizer_parameters={'complexity':[2,3,4]}
    n_iter=20
    estimator_parameters={'n_iter':p_randint(5, 200, size=n_iter),
                          'penalty':['l1','l2','elasticnet'],
                          'l1_ratio':p_uniform(0.1,0.9, size=n_iter),
                          'loss':['log'],
                          'power_t':p_uniform(0.1, size=n_iter),
                          'alpha': [10**x for x in range(-8,0)],
                          'eta0': [10**x for x in range(-4,-1)],
                          'learning_rate': ["invscaling", "constant", "optimal"],
                          'n_jobs':[n_jobs]}

    model.optimize(list(iterable_pos_train), list(iterable_neg_train),
                   max_total_time=60*10,
                   score_func=lambda avg_score,std_score : avg_score - std_score * 2,
                   scoring='roc_auc',
                   n_iter=n_iter,
                   pre_processor_parameters=pre_processor_parameters,
                   vectorizer_parameters=vectorizer_parameters,
                   estimator_parameters=estimator_parameters)
    return model.get_estimator()
'''


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
    '''
    param_dist = {"n_iter": randint(5, 25),
              "power_t": uniform(0.1),
              "alpha": uniform(1e-05, 1e-01),
              "eta0": uniform(1e-04, 1e-01),
              "loss":[ 'log'],
              "penalty": ["l1", "l2", "elasticnet"],
              "learning_rate": ["invscaling", "constant", "optimal"]}
    '''

    scoring = 'roc_auc'
    # scoring = metrics.brier_score_loss dowsnt work..
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



