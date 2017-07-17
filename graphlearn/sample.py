#!/usr/bin/env python

"""Provides the sampler class."""

import logging
logger = logging.getLogger(__name__)


class Sampler(object):

    def __init__(self, grammar=None, score_estimator=None, n_steps=3):
        self.lsgg = grammar
        if len(self.lsgg.productions) == 0:
            raise Exception("sampler needs a trained grammar")

        self.n_steps = n_steps
        self.score_estimator = score_estimator

    def fit(self):
        pass

    def transform(self, graph):
        logger.log(5, '\n\nsample: start transformation')
        for self.step in range(0, self.n_steps):
            graph, score = self.choosenext(graph)
            yield graph, score

    def choosenext(self, graph):
        proposals = list(self.lsgg.neighbors(graph))
        logger.log(5, 'sample: proposalcount %d' % len(proposals))
        return max(self.score(proposals), key=lambda x: x[1])

    def score(self, graphs):
        scores = self.score_estimator.decision_function(graphs)[0]
        logger.log(5, 'sample: score scores: %s' % str(scores))
        return zip(graphs, scores)
