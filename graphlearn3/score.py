#!/usr/bin/env python

"""Provides the wrapper for estimators."""

from eden.graph import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

class RandomEstimator():
    def __init__(self):
        pass
    def fit(self, graph=None, vectorizer=Vectorizer()):
        return self

    def decision_function(self, graphs):
        return np.array(  [random.random() for e in range(len(graphs))])