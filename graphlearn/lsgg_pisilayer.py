
import structout as so 

from graphlearn.test import transformutil
import copy
import functools
from graphlearn import local_substitution_graph_grammar
from graphlearn import lsgg_core_interface_pair
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import logging
logger = logging.getLogger(__name__)
from graphlearn import lsgg_layered 
from graphlearn import lsgg_pisi


class lsgg_pisilayer(lsgg_pisi.PiSi, lsgg_layered.lsgg_layered):

    def _make_cip(self, core=None, graph=None):

        basecip = lsgg_core_interface_pair.make_cip(root_node=core,
                                                    graph=graph,
                                                    radius=radius,
                                                    thickness=thickness)


        base_thickness = 2*self.decomposition_args['base_thickness']
        pisi_thickness = 2*self.decomposition_args['thickness_pisi']





        # edge to vertex Basegraph
        orig_graph = graph.cip_graph['original']
        expanded_orig_graph = lsgg_core_interface_pair._edge_to_vertex(orig_graph)

        # make a copy, collapse core
        expanded_orig_graph_collapsed =  expanded_orig_graph.copy()
        nodes_in_core = list (functools.reduce(lambda x,y: x|y, [basecip.cip_graph.nodes[i]['contracted']
                                                                 for i in basecip.core_nodes if 'edge' not in basecip.cip_graph.nodes[i]]))

        edges_in_core = [n for n,d in expanded_orig_graph_collapsed.nodes(data=True)
                             if 'edge' in d and all([z in nodes_in_core for z in expanded_orig_graph_collapsed.neighbors(n) ])]
        for n in nodes_in_core[1:]+edges_in_core:
            transformutil.merge_edge(expanded_orig_graph_collapsed, nodes_in_core[0], n)


        # distances...
        dist = nx.single_source_shortest_path_length(
                                expanded_orig_graph_collapsed,
                                nodes_in_core[0], pisi_thickness)

        # set distance dependant label
        lsgg_core_interface_pair._add_hlabel(expanded_orig_graph)
        ddl = 'distance_dependent_label'
        for id, dst in dist.items():
            if dst>0:
                expanded_orig_graph.nodes[id][ddl] = expanded_orig_graph.nodes[id]['hlabel'] + dst



        basecip.interface_nodes = [id for id, dst in dist.items()
                   if 0 < dst <= base_thickness]
        interface_hash = lsgg_core_interface_pair.graph_hash(expanded_orig_graph.subgraph(basecip.interface_nodes))
        basecip.interface_graph = expanded_orig_graph.subgraph(basecip.interface_nodes).copy()
        basecip.core_nodes=nodes_in_core+edges_in_core
        basecip.interface_hash =  hash((interface_hash,basecip.interface_hash))
        basecip.cip_graph= expanded_orig_graph.subgraph(basecip.interface_nodes + nodes_in_core + edges_in_core).copy()

        # do the pisi stuff
        pisi_nodes = [id for id, dst in dist.items() if 1 < dst <= pisi_thickness  ]
        if len(pisi_nodes) == 0: 
            logger.log(10,'skipping because interface empty')
            return None
        pisi_graph = expanded_orig_graph.subgraph(pisi_nodes).copy()
        basecip.pisi_hash = {lsgg_core_interface_pair.graph_hash(pisi_graph)}
        basecip.pisi_vectors = lsgg_core_interface_pair.eg.vectorize([pisi_graph])
        return basecip



