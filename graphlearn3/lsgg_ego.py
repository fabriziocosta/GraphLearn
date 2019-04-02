from graphlearn3 import lsgg
from graphlearn3 import lsgg_cip
from graphlearn3.test import transformutil
import networkx as nx
import copy
import functools
import logging
from ego.component import convert 
logger = logging.getLogger(__name__)

"""We adjust lsgg_layered such that it works with EGO decomposition"""

class lsgg_ego(lsgg.lsgg):
    def _roots(self,graph): 
        ego_decomp = self.decomposition_function(convert(graph))
        return [list(e) for e in ego_decomp[1]] # i should hack the edges in here... where is an example for edge decomp?

    def __init__(self,decomposition_function, preprocessor=lambda x: x, **kwargs):
        self.preprocessor = preprocessor
        self.decomposition_function = decomposition_function
        return super().__init__(**kwargs)
    
    def _cip_extraction(self,graph):
        graph = self.preprocessor(graph.copy())
        return super()._cip_extraction(graph)

    def _neighbors_given_cips(self, graph, orig_cips):
        graph = self.preprocessor(graph) 
        return super()._neighbors_given_cips(graph,orig_cips)

########################3
#  TESTS
###########################

def test_lsgg_ego_nodedecomp(out=False):
    # node decompo is a decomposer that returns sets of nodes
    from graphlearn3.util import util as util_top
    from ego.cycle_basis import decompose_cycles_and_non_cycles
    from structout import gprint 
    decomposition_args={ 'radius_list': [0], 
                        "thickness_list": [1]}

    lsggg = lsgg_ego(decomposition_args=decomposition_args,
                    decomposition_function= decompose_cycles_and_non_cycles)

    g = util_top.test_get_circular_graph()
    gplus=g.copy()
    gplus.node[0]['label']='weird'
    lsggg.fit([g, gplus, g,gplus])
    stuff= lsggg.neighbors(gplus).__next__()
    if out: gprint(stuff)
    assert len(stuff) == 8

def test_lsgg_ego_edgedecomp(out=False):
    pass 
    # edge decompo is a decomposer that returns sets of edges
    from graphlearn3.util import util as util_top
    from ego.cycle_basis import decompose_cycles_and_non_cycles
    from structout import gprint 
    decomposition_args={ 'radius_list': [0], 
                        "thickness_list": [1]}

    lsggg = lsgg_ego(decomposition_args=decomposition_args,
                    decomposition_function= decompose_cycles_and_non_cycles)

    g = util_top.test_get_circular_graph()
    gplus=g.copy()
    gplus.node[0]['label']='weird'
    lsggg.fit([g, gplus, g,gplus])
    stuff= lsggg.neighbors(gplus).__next__()
    if out: gprint(stuff)
    assert len(stuff) == 8

def test_lsgg_ego_preprocessor(out=False):
    # edge decompo is a decomposer that returns sets of edges
    from graphlearn3.util import util as util_top
    from ego.cycle_basis import decompose_cycles_and_non_cycles
    from structout import gprint 
    from structout.graph import ginfo
    from ego.abstract_label import preprocess_abstract_label
    decomposition_args={ 'radius_list': [0], 
                        "thickness_list": [1]}

    lsggg = lsgg_ego(decomposition_args=decomposition_args,
                    decomposition_function= decompose_cycles_and_non_cycles,
                    preprocessor = preprocess_abstract_label(node_label='C', edge_label='1') )

    g = util_top.test_get_circular_graph()
    gplus=g.copy()
    gplus.node[0]['label']='weird'
    lsggg.fit([g, gplus, g,gplus])
    stuff= lsggg.neighbors(gplus).__next__()
    if out: gprint(stuff)
    assert len(stuff) == 5

def demo_lsgg_ego():
    from graphlearn3.util import util as util_top
    from ego.cycle_basis import decompose_cycles_and_non_cycles
    from structout import gprint 

    decomposition_args={ 'radius_list': [0], 
                        "thickness_list": [1]}
    lsggg = lsgg_ego(decomposition_args=decomposition_args,
                    decomposition_function= decompose_cycles_and_non_cycles)

    g = util_top.get_cyclegraphs()
    for gg in g:
        gprint(gg)
        for cip in lsggg._cip_extraction(gg):
            gprint(cip.graph,color=[cip.interface_nodes,cip.core_nodes] )
        print("#"*80)


