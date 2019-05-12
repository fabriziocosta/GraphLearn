from graphlearn3 import lsgg
from graphlearn3 import lsgg_cip
from graphlearn3.test import transformutil
import networkx as nx
import copy
import functools
import logging
from ego.component import convert
from ego.encode import make_encoder

logger = logging.getLogger(__name__)

"""We adjust lsgg_layered such that it works with EGO decomposition"""


class lsgg_ego(lsgg.lsgg):

    def _roots(self,graph):
        codes, ego_decomp_fragments = self.decomposition_function(graph)
        #import structout as so
        #print(dir(ego_decomp_fragments[0]))
        #so.graph.ginfo(ego_decomp_fragments[0])
        return ego_decomp_fragments


    def __init__(self,decomposition_function, **kwargs):
        self.decomposition_function = make_encoder(decomposition_function, bitmask=2**20-1)
        return super().__init__(**kwargs)


    def _extract_core_and_interface(self, root_node=None, graph=None, radius=None, thickness=None):
        assert radius ==0, "musst be zero because we dont expand cores here."
        # root_node is a subgraph that we use as core (because we said so in _roots()
        # to avoid node id misstranslations in the expanded graph, i mark all the elements
        # as core before expanding
        for n in root_node.nodes:
            graph.node[n]['core'] = True
        for a,b in root_node.edges:
            graph[a][b]['core'] = True

        egraph = lsgg_cip._edge_to_vertex(graph)

        for n in root_node.nodes:
            graph.node[n].pop('core')
        for a,b in root_node.edges:
            graph[a][b].pop('core')

        graph = egraph
        lsgg_cip._add_hlabel(graph)

        core_nodes = [index for index, dict in graph.nodes.data() if 'core' in dict]
        dist = {a:b for (a,b) in lsgg_cip.short_paths(graph, core_nodes,
                                         thickness+radius)
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

########################3
#  TESTS
###########################

def test_lsgg_ego_nodedecomp(out=False):
    # node decompo is a decomposer that returns sets of nodes
    from graphlearn3.util import util as util_top
    from structout import gprint 
    from ego.decompose import  compose , concatenate
    from ego.paired_neighborhoods import  decompose_neighborhood
    from ego.pair import decompose_pair
    z = compose(decompose_pair(distance=2),  decompose_neighborhood(radius=1))
    decomposition_args={ 'radius_list': [0], 
                        "thickness_list": [1]}

    lsggg = lsgg_ego(decomposition_args=decomposition_args,
                    decomposition_function= z)

    g = util_top.test_get_circular_graph()
    gplus=g.copy()
    gplus.node[0]['label']='weird'
    lsggg.fit([g, gplus, g,gplus])
    stuff= lsggg.neighbors(gplus).__next__()
    if out: gprint(stuff)
    assert len(stuff) == 8


def test_lsgg_ego_edgedecomp(out=False):
    # edge decompo is a decomposer that returns sets of edges
    from graphlearn3.util import util as util_top
    from ego.path import decompose_path
    from structout import gprint 
    from structout.graph import ginfo
    decomposition_args={ 'radius_list': [0], 
                        "thickness_list": [1]}
    
    class testego(lsgg_ego): # visualises the substitution
        def _core_substitution(self, graph, cip, cip_):
            graphs = [cip.graph,cip_.graph]
            gprint(graphs, color=[[cip.core_nodes], [cip_.core_nodes] ])
            return super()._core_substitution(graph,cip,cip_)

    lsggg = testego(decomposition_args=decomposition_args,
            decomposition_function= lambda x : decompose_path(x,min_len =1, max_len = 4) ) 

    g = util_top.test_get_circular_graph()
    gplus=g.copy()
    gplus.node[0]['label']='weird'

    lsggg.fit([g, gplus, g,gplus])
    stuff= lsggg.neighbors(gplus).__next__()
    if out: gprint(gplus)
    if out: gprint(stuff)
    assert len(stuff) == 8



