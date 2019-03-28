from graphlearn3 import lsgg
from graphlearn3 import lsgg_cip
from graphlearn3.test import transformutil
import networkx as nx
import copy
import functools
import logging
logger = logging.getLogger(__name__)


# NX 2.2 hides this important function from the user
from networkx.algorithms.shortest_paths.unweighted import _single_shortest_path_length as short_paths


"""We adjust lsgg_layered such that it works with EGO"""

class lsgg_ego(lsgg.lsgg):

 
    def _nodes(self,graph): # _nodes is basically a list of extraction elements
        return self.decomposition_args["decompose_func"](graph)

    def _extract_core_and_interface(self,root_node=None,
                                   graph=None,
                                   radius=None,
                                   thickness=None):
        '''

        Parameters
        ----------
        root_node
        graph
        radius
        thickness

        Returns
        -------
            makes a cip oO
        '''
        # preprocessing
        graph = lsgg_cip._edge_to_vertex(graph)
        lsgg_cip._add_hlabel(graph)
        dist = {a:b for (a,b) in short_paths(graph, root_node, thickness)}

        # find interesting nodes:
        core_nodes = root_node 
        interface_nodes = [id for id, dst in dist.items() if dst > 0]

        return lsgg_cip._extract_core_and_interface(root_node,graph,radius,thickness,dist,core_nodes,interface_nodes)




def test_lsgg_ego():
    from graphlearn3.util import util as util_top
    from ego.cycle_basis import decompose_cycles_and_non_cycles
    from ego.component import convert 
    decomposition_args={ 
                        'radius_list': [0], # musst be just [0]
                        "decompose_func": lambda x: [list(e) for e in   decompose_cycles_and_non_cycles(convert(x))[1]],
                        #"decompose_func": lambda x:[ [n] for n in x.nodes()],
                        "thickness_list": [1]}
                        #'hash_bitmask': 2**20-1}

    lsggg = lsgg_ego(decomposition_args=decomposition_args)

    # ...
    from structout import gprint 
    g = util_top.get_cyclegraphs()
    for gg in g:
        gprint(gg)
        for cip in lsggg._cip_extraction(gg):
            gprint(cip.graph,color=[cip.interface_nodes,cip.core_nodes] )
        print("#"*80)







