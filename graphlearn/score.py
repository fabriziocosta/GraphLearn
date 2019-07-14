#!/usr/bin/env python

"""Provides the wrapper for estimators."""

from eden.graph import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM
import random
import numpy as np
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
    
    def __init__(self,model=None):
        if not model: 
            self.model = OneClassSVM()
        else:
            self.model=model

    def fit(self,graphs, vectorizer=Vectorizer()):
        self.model.fit(vectorizer.transform(graphs) )
        self.vectorizer = vectorizer
        return self

    def decision_function(self, graphs):
        vecs = self.vectorizer.transform(graphs)
        return self.model.score_samples(vecs)

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

