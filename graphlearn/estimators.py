#!/usr/bin/env python

"""Provides the wrapper for estimators."""

from eden.graph import Vectorizer
from sklearn.metrics.pairwise import cosine_similarity


class simple_directed_estimator():

    def __init__(self, graph=None, vectorizer=Vectorizer(n_jobs=1)):
        self.reference_vec = vectorizer.transform([graph])
        self.vectorizer = vectorizer

    def decision_function(self, graphs):
        vecs = self.vectorizer.transform(graphs)
        return cosine_similarity(self.reference_vec, vecs)
