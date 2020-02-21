#!/usr/bin/env python
"""Provides scikit interface."""
import graphlearn.sample
from graphlearn import local_substitution_graph_grammar
from graphlearn import lsgg_core_interface_pair as cip
import networkx as nx
import logging
from ego.encode import make_encoder

logger = logging.getLogger(__name__)

"""We adjust lsgg_layered such that it works with EGO decomposition"""


class lsgg_ego(graphlearn.sample.LocalSubstitutionGraphGrammarSample):



    def _ego_node_fix(self,graph, core): 
        graph= cip._edge_to_vertex(graph)
        id_dist = { n: di for (n,di) in cip.short_paths(graph,core.nodes(),1)}
        return graph.subgraph(cip.get_node_set(id_dist,0,graph))

    def _get_cores(self, graph):
        codes, ego_decomp_fragments = self.decomposition_function(graph)
        
        return [self._ego_node_fix(graph, core) for core in ego_decomp_fragments]

    def __init__(self, decomposition_function, **kwargs):
        self.decomposition_function = make_encoder(decomposition_function, bitmask=2**20 - 1)
        super(lsgg_ego, self).__init__(**kwargs)


    def root_neighbors(self, graph, roots, n_neighbors=1000):
        """root_neighbors."""
        cores = self._get_cores(graph)
        for core in cores:
            if any([n in roots for n in core.nodes()]): # intersection between potential cores and allowed roots
                for neigh in self.neighbors_root(graph,core):
                    if n_neighbors > 0:
                        yield neigh
                        n_neighbors = n_neighbors -1
                    else:
                        return

        '''
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
        '''
