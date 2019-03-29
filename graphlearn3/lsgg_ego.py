from graphlearn3 import lsgg
from graphlearn3 import lsgg_cip
from graphlearn3.test import transformutil
import networkx as nx
import copy
import functools
import logging
logger = logging.getLogger(__name__)


# NX 2.2 hides this important function from the user
# from networkx.algorithms.shortest_paths.unweighted import _single_shortest_path_length as short_paths


"""We adjust lsgg_layered such that it works with EGO decomposition"""

class lsgg_ego(lsgg.lsgg):
    def _roots(self,graph): 
        return self.decomposition_args["decompose_func"](graph)

  


def demo_lsgg_ego():
    from graphlearn3.util import util as util_top
    from ego.cycle_basis import decompose_cycles_and_non_cycles
    from ego.component import convert 
    from structout import gprint 

    decomposition_args={ 
                        'radius_list': [0], # musst be just [0]
                        "decompose_func": lambda x: [list(e) for e in   decompose_cycles_and_non_cycles(convert(x))[1]],
                        # this also works here but basically does nothing
                        #"decompose_func": lambda x:[ [n] for n in x.nodes()],
                        "thickness_list": [1]}

    lsggg = lsgg_ego(decomposition_args=decomposition_args)

    g = util_top.get_cyclegraphs()
    for gg in g:
        gprint(gg)
        for cip in lsggg._cip_extraction(gg):
            gprint(cip.graph,color=[cip.interface_nodes,cip.core_nodes] )
        print("#"*80)



def test_lsgg_ego():
    from graphlearn3.util import util as util_top
    from ego.cycle_basis import decompose_cycles_and_non_cycles
    from ego.component import convert 
    from structout import gprint 
    decomposition_args={ 
                        'radius_list': [0], # musst be just [0]
                        "decompose_func": lambda x: [list(e) for e in   decompose_cycles_and_non_cycles(convert(x))[1]],
                        "thickness_list": [1]}

    lsggg = lsgg_ego(decomposition_args=decomposition_args)
    g = util_top.test_get_circular_graph()
    gplus=g.copy()
    gplus.node[0]['label']='weird'

    lsggg.fit([g, gplus, g,gplus])
    stuff= lsggg.neighbors(gplus).__next__()
    gprint(stuff)
    assert len(stuff) == 8


