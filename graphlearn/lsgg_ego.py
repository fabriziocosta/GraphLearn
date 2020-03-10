#!/usr/bin/env python
"""Provides scikit interface."""
import graphlearn.sample
from graphlearn import lsgg_core_interface_pair as cip
import logging
from ego.encode import make_encoder

logger = logging.getLogger(__name__)

"""We adjust lsgg_layered such that it works with EGO decomposition"""


class lsgg_ego(graphlearn.sample.LocalSubstitutionGraphGrammarSample):

    '''
    def _ego_node_fix(self, graph, core):
        id_dist = {n: di for (n, di) in cip.short_paths(graph, core.nodes(), 1)}
        return graph.subgraph(cip.get_node_set(id_dist, 0, graph))
    '''

    def _get_cores(self, graph):
        codes, ego_decomp_fragments = self.encoder(graph)
        #graph = cip._edge_to_vertex(graph)
        return  ego_decomp_fragments

    def set_decomposition(self, decomposition):
        self.encoder = make_encoder(decomposition, bitmask=2**20 - 1)

    def __init__(self, decomposition_function, **kwargs):
        self.set_decomposition(decomposition_function)
        super(lsgg_ego, self).__init__(**kwargs)

    def root_neighbors(self, graph, roots, n_neighbors=1000):
        """root_neighbors. roots are some nodes from graph, a core musst intersect with root"""
        cores = self._get_cores(graph)
        for core in cores:
            if any([n in roots for n in core.nodes()]):  # intersection between potential cores and allowed roots
                for neigh in self.neighbors_core(graph, core):
                    if n_neighbors > 0:
                        yield neigh
                        n_neighbors = n_neighbors - 1
                    else:
                        return
