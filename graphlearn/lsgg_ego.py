#!/usr/bin/env python
"""Provides scikit interface."""
import graphlearn.sample
from graphlearn import local_substitution_graph_grammar
from graphlearn import lsgg_core_interface_pair
import networkx as nx
import logging
from ego.encode import make_encoder

logger = logging.getLogger(__name__)

"""We adjust lsgg_layered such that it works with EGO decomposition"""


class lsgg_ego(graphlearn.sample.LocalSubstitutionGraphGrammarSample):

    def _get_cores(self, graph):
        codes, ego_decomp_fragments = self.decomposition_function(graph)
        return ego_decomp_fragments

    def __init__(self, decomposition_function, **kwargs):
        self.decomposition_function = make_encoder(decomposition_function, bitmask=2**20 - 1)
        super(lsgg_ego, self).__init__(**kwargs)


    def root_neighbors(self, graph, roots, n_neighbors=1000):
        """root_neighbors."""
        try:

            for root in roots:
                for root_graph in self._get_cores(graph):
                    if root not in list(root_graph.nodes()):
                        continue

                    cip = self._make_cip(graph=graph)
                    n_neighbors_counter = n_neighbors

                    for congruent_cip in self._get_congruent_cips(cip):
                        neighbor = self._substitute_core(graph, cip, congruent_cip)
                        if n_neighbors_counter > 0:
                            n_neighbors_counter = n_neighbors_counter - 1
                            yield neighbor
                        else:
                            raise StopIteration
        except StopIteration:
            return
