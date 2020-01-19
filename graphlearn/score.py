#!/usr/bin/env python

"""Provides the wrapper for estimators."""

from eden.graph import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM
import random
import numpy as np
from graphlearn.util.multi import mpmap
import scipy as sp
import logging 
logger = logging.getLogger(__name__)

class SimpleDistanceEstimator():
    def __init__(self):
        self.reference_vec, self.vectorizer = None, None

    def fit(self, graph, vectorizer=Vectorizer()):
        self.reference_vec = vectorizer.transform([graph])
        self.vectorizer = vectorizer
        return self

    def decision_function(self, graphs):
        vecs = self.vectorizer.transform(graphs)
        return cosine_similarity(self.reference_vec, vecs)[0]

class OneClassEstimator():
    
    def __init__(self,model=None, n_jobs=1,vectorizer=Vectorizer()):
        if not model: 
            self.model = OneClassSVM(gamma='auto')
        else:
            self.model=model
        self.n_jobs=n_jobs
        self.vectorizer=vectorizer
    
    def transform(self,graphs):
        if self.n_jobs==1:
            return self.vectorizer.transform(graphs)
        else: 
            return sp.sparse.vstack( mpmap( self.vectorizer.transform, [[g] for g in graphs], poolsize=self.n_jobs ) ) 

    def fit(self,graphs):
        self.model.fit(self.transform(graphs) )
        return self

    def decision_function(self, graphs):
        vecs = self.transform(graphs)
        return self.model.decision_function(vecs)

class OneClassSizeHarmMean(OneClassEstimator):
    def fit(self,graphs):
        le = np.array(list(map(len,graphs)))
        #self.sizedist = sp.stats.norm(loc=le.mean(),scale=le.std()/2)
        self.size_mean=le.mean()
        self.size_std = le.std()
        logger.log(5,f"sizedist: mean{le.mean()}, scale {le.std()}")
        super().fit(graphs)
        return self

    def decision_function(self,graphs):
        vecs = self.transform(graphs)
        scores =  self.model.decision_function(vecs)
        scores2 = [sp.stats.logistic.cdf(a,0,1) for a in scores]
        norm = lambda x:  np.exp(-(((len(x)-self.size_mean)/self.size_std)**2)*.5) 
        sizefac = [ norm(x) if len(x)> self.size_mean else 1  for x in graphs ]
        #sizefac2 = [sp.stats.logistic.cdf(-a,-2,1) for a in sizefac]  
        res= [ sp.stats.hmean((a,b))  for a,b in zip(sizefac,scores2)  ]
        logger.log(5,f"svm:   {scores} -> {scores2}")
        logger.log(5,f"size:  {[len(x) for x in graphs]} -> {sizefac}")
        logger.log(5,f"hmean: {res}")
        return res

class OneClassAndSizeFactor(OneClassEstimator):
    def decision_function(self,graphs):
        if 'sizefactor' not in self.__dict__:
            print ("OneClassAndSizeFactor has no size factor")
        vecs = self.transform(graphs)
        return self.model.decision_function(vecs)*np.array([self.sizepen(x) for x in graphs])
    def sizepen(self,g):
        diff =  abs(len(g) - self.sizefactor)
        return 1 - diff*self.sizepenalty

class RandomEstimator():
    def __init__(self):
        pass
    def fit(self, graph=None, vectorizer=Vectorizer()):
        return self

    def decision_function(self, graphs):
        return np.array(  [random.random() for e in range(len(graphs))])



def internal_tessssst_oneclass():
    # python -c "import score as s; s.internal_tessssst_oneclass()"
    # lets get sum data
    from toolz import curry, pipe
    from eden_chem.io.pubchem import download
    from eden_chem.io.rdkitutils import sdf_to_nx
    download_active = curry(download)(active=True)
    download_inactive = curry(download)(active=False)
    def get_pos_graphs(assay_id): return pipe(assay_id, download_active, sdf_to_nx, list)
    assay_id='624249'
    gr = get_pos_graphs(assay_id)
    est = OneClassEstimator().fit(gr)
    print (est.decision_function(gr))

