#!/usr/bin/env python
"""Provides scikit interface."""

from graphlearn3 import lsgg
from graphlearn3 import lsgg_cip
import networkx as nx
import logging
from ego.encode import make_encoder

logger = logging.getLogger(__name__)

"""We adjust lsgg_layered such that it works with EGO decomposition"""


class lsgg_ego(lsgg.lsgg):

    def _roots(self, graph):
        codes, ego_decomp_fragments = self.decomposition_function(graph)
        return ego_decomp_fragments

    def __init__(self, decomposition_function, **kwargs):
        self.decomposition_function = make_encoder(decomposition_function, bitmask=2**20 - 1)
        return super(lsgg_ego, self).__init__(**kwargs)

    def _extract_core_and_interface(self, root_node=None, graph=None, radius=None, thickness=None):
        assert radius == 0, "musst be zero because we dont expand cores here."
        # root_node is a subgraph that we use as core (because we said so in _roots()
        # to avoid node id misstranslations in the expanded graph, i mark all the elements
        # as core before expanding
        for n in root_node.nodes:
            graph.node[n]['core'] = True
        for a, b in root_node.edges:
            graph[a][b]['core'] = True

        egraph = lsgg_cip._edge_to_vertex(graph)

        for n in root_node.nodes:
            graph.node[n].pop('core')
        for a, b in root_node.edges:
            graph[a][b].pop('core')

        graph = egraph
        lsgg_cip._add_hlabel(graph)

        core_nodes = [index for index, dict in graph.nodes.data() if 'core' in dict]
        dist = {a: b for (a, b) in lsgg_cip.short_paths(graph, core_nodes,
                                                        thickness + radius)
                }

        interface_nodes = [id for id, dst in dist.items()
                           if radius < dst <= radius + thickness]

        # calculate hashes
        core_hash = lsgg_cip.graph_hash_core(graph.subgraph(core_nodes))
        node_name_label = lambda id, node: node['hlabel'] + dist[id] - radius
        interface_hash = lsgg_cip.graph_hash(graph.subgraph(interface_nodes),
                                             node_name_label=node_name_label)

        # copy cip and mark core/interface
        cip_graph = graph.subgraph(core_nodes + interface_nodes).copy()
        ddl = 'distance_dependent_label'
        for no in interface_nodes:
            cip_graph.node[no][ddl] = cip_graph.node[no]['hlabel'] + dist[no] - (radius + 1)

        interface_graph = nx.subgraph(cip_graph, interface_nodes)

        return lsgg_cip.CoreInterfacePair(interface_hash,
                                          core_hash,
                                          cip_graph,
                                          radius,
                                          thickness,
                                          len(core_nodes),
                                          root=root_node,
                                          core_nodes=core_nodes,
                                          interface_nodes=interface_nodes,
                                          interface_graph=interface_graph)
