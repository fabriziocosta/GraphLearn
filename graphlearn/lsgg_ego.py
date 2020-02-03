#!/usr/bin/env python
"""Provides scikit interface."""

from graphlearn import local_substitution_graph_grammar
from graphlearn import lsgg_core_interface_pair
import networkx as nx
import logging
from ego.encode import make_encoder

logger = logging.getLogger(__name__)

"""We adjust lsgg_layered such that it works with EGO decomposition"""


class lsgg_ego(local_substitution_graph_grammar.LocalSubstitutionGraphGrammarSample):

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



    def _make_cip(self, core=None, graph=None):

        # root_node is a subgraph that we use as core (because we said so in _roots()
        # to avoid node id misstranslations in the expanded graph, i mark all the elements
        # as core before expanding
        for n in core.nodes:
            graph.nodes[n]['core'] = True
        for a, b in core.edges:
            graph[a][b]['core'] = True

        egraph = lsgg_core_interface_pair._edge_to_vertex(graph)

        for n in core.nodes:
            graph.nodes[n].pop('core')
        for a, b in core.edges:
            graph[a][b].pop('core')

        graph = egraph
        lsgg_core_interface_pair._add_hlabel(graph)

        core_nodes = [index for index, dict in graph.nodes.data() if 'core' in dict]
        dist = {a: b for (a, b) in lsgg_core_interface_pair.short_paths(graph, core_nodes,
                                                                        self.thickness )
                }

        interface_nodes = [id for id, dst in dist.items()
                           if 0 < dst <=  self.thickness]

        # calculate hashes
        core_hash = lsgg_core_interface_pair.graph_hash(graph.subgraph(core_nodes))
        node_name_label = lambda id, node: node['hlabel'] + dist[id]
        interface_hash = lsgg_core_interface_pair.graph_hash(graph.subgraph(interface_nodes),
                                                             get_node_label=node_name_label)

        # copy cip and mark core/interface
        cip_graph = graph.subgraph(core_nodes + interface_nodes).copy()
        ddl = 'distance_dependent_label'
        for no in interface_nodes:
            cip_graph.nodes[no][ddl] = cip_graph.nodes[no]['hlabel'] + dist[no] - 1

        interface_graph = nx.subgraph(cip_graph, interface_nodes)

        return lsgg_core_interface_pair.CoreInterfacePair(interface_hash,
                                                          core_hash,
                                                          cip_graph,
                                                          0,
                                                          self.thickness,
                                                          len(core_nodes),
                                                          root=core,
                                                          core_nodes=core_nodes,
                                                          interface_nodes=interface_nodes,
                                                          interface_graph=interface_graph)
