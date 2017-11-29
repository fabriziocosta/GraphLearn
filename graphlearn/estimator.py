#!/usr/bin/env python

"""Provides the wrapper for estimators."""

from eden.graph import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity


class simpleDirectedEstimator():
    def __init__(self):
        self.reference_vec, self.vectorizer = None, None

    def fit(self, graph, vectorizer=Vectorizer()):
        self.reference_vec = vectorizer.transform([graph])
        self.vectorizer = vectorizer
        return self

    def decision_function(self, graphs):
        vecs = self.vectorizer.transform(graphs)
        return cosine_similarity(self.reference_vec, vecs)
