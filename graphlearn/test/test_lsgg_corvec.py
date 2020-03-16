#!/usr/bin/env python


"""Provides the graph grammar class."""

#from graphlearn import lsgg_core_interface_pair as lcip
#from graphlearn import local_substitution_graph_grammar as lsgg


from graphlearn.util import util
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
import networkx as nx
import numpy as np
from graphlearn import cipcorevector as ccv 
import logging
logger = logging.getLogger(__name__)
import structout as so
def test_gtrain():

    lsgg = ccv.LsggCoreVec(decompose_neighborhood)
    g = nx.path_graph(5)
    g.add_edge(2,4)
    g = util._edenize_for_testing(g)

    cores = lsgg._get_cores(g)
    cip = lsgg._make_cip(cores[0], g)

    assert cip.core_vec.sum()==2 



#############
# ok we need a proper test
##############3


# make an estimator, with chem graphs
from  graphlearn.test.test_pisi import getgraphs
import ego.vectorize as evec

def decomp(egostuff):
    # i expect this to work,, but probably ego is bugged
    #return [ lambda x: decompose_neighborhood(x,radius=y) for y in [0,1,2]]
    egostuff = list(egostuff)
    return [e for radius in [1,2] for e in decompose_neighborhood(egostuff,radius=radius)]

decomp = decompose_neighborhood

def vectorize(graphs):
    return evec.vectorize(graphs, decomposition_funcs = decomp) 

from sklearn.svm import OneClassSVM as svm

def get_esti():
    graphs =  getgraphs()[:100]
    vectors = vectorize(graphs)
    return svm(kernel='linear').fit(vectors)






def selectbest_TOOOLD(cips_, cip, esti, n=2):
    if len(cips_) < n: 
        return cips_
    cvec = np.vstack ([c.core_vec for c in cips_ ])
    print("cvec:", cvec.shape)
    ranks =  np.argsort(esti.decision_function(cvec)) 
    return [cips_[x] for x in ranks[-n:]]



def selectbest(my_other, esti, n=2):

    if len(my_other) < n: 
        return my_other
    cvec = np.vstack ([other.core_vec - my.core_vec for my,other in my_other ])
    print("cvec:", cvec.shape)
    ranks =  np.argsort(esti.decision_function(cvec)) 
    best=  [my_other[x] for x in ranks[-n:]]
    so.gprint([c.graph for c in best[0]])
    so.gprint([c.graph for c in best[1]])
    return best

    

def test_corvec(): 
    
    # ok so wefit a grammar 
    lsgg = ccv.LsggCoreVec(decomp, selectbest)
    graphs = getgraphs()
    lsgg.fit(graphs[:100])
    esti = get_esti()
    # ok we should be able to do this now....
    so.gprint(list(lsgg.neighbors(graphs[3], [esti,2])))

