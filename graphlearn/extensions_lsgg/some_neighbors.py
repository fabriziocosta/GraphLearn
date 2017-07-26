#!/usr/bin/env python

"""Provides a modified graph grammar class."""

import random
import graphlearn.lsgg as base_grammar


class lsgg(base_grammar.lsgg):

    def neighbors_sample(self, graph, n_neighbors):
        for root in random.sample(graph.nodes(), len(graph)):
            cips = self._rooted_decompose(graph, root)
            for neighbor in self._neighbors_given_cips(graph, cips):
                if n_neighbors > 0:
                    n_neighbors -= 1
                    yield neighbor
                else:
                    raise StopIteration
