from graphlearn.lsgg_ego import lsgg_ego
import logging

logger = logging.getLogger(__name__)



'''
def test_lsgg_ego_nodedecomp(out=False):
    # node decompo is a decomposer that returns sets of nodes
    from graphlearn.util import util as util_top
    from structout import gprint
    from ego.decompose import compose, concatenate
    from ego.paired_neighborhoods import decompose_neighborhood
    from ego.pair import decompose_pair
    z = compose(decompose_pair(distance=2), decompose_neighborhood(radius=1))
    decomposition_args = {'radius_list': [0],
                          "thickness": 1}

    lsggg = lsgg_ego(decomposition_args=decomposition_args,
                     decomposition_function=z)

    g = util_top.test_get_circular_graph()
    gplus = g.copy()
    gplus.nodes[0]['label'] = 'weird'
    lsggg.fit([g, gplus, g, gplus])
    stuff = lsggg.neighbors(gplus).__next__()
    if out:
        gprint(stuff)
    assert len(stuff) == 8


def test_lsgg_ego_edgedecomp(out=False):
    # edge decompo is a decomposer that returns sets of edges
    from graphlearn.util import util as util_top
    from ego.path import decompose_path
    from structout import gprint
    from structout.graph import ginfo
    decomposition_args = {'radius_list': [0],
                          "thickness": 1}

    class testego(lsgg_ego):  # visualises the substitution

        def _substitute_core(self, graph, cip, cip_):
            graphs = [cip.cip_graph, cip_.cip_graph]
            gprint(graphs, color=[[cip.core_nodes], [cip_.core_nodes]])
            return super()._core_substitution(graph, cip, cip_)

    lsggg = testego(decomposition_args=decomposition_args,
                    decomposition_function=lambda x: decompose_path(x, min_len=1, max_len=4))

    g = util_top.test_get_circular_graph()
    gplus = g.copy()
    gplus.nodes[0]['label'] = 'weird'

    lsggg.fit([g, gplus, g, gplus])
    stuff = lsggg.neighbors(gplus).__next__()
    if out:
        gprint(gplus)
    if out:
        gprint(stuff)
    assert len(stuff) == 8
'''







from graphlearn.util import util
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
import networkx as nx

import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=5)

def test_ego():
    
    # init grpah and grammar 
    g = nx.path_graph(5)
    #g.add_edge(2,4)
    g = util._edenize_for_testing(g)
    lsgg = lsgg_ego(decompose_neighborhood, thickness=1,nodelevel_radius_and_thickness=False)

    # make a cip
    cores = lsgg._get_cores(g)
    cip = lsgg._get_cip(cores[0], g)
    print(cip.ascii())
    
    # train a grammar
    lsgg.fit([g,g,g])
    lsgg.structout()


    # generate neighs
    s = list(lsgg.neighbors(g))
    import structout as so
    so.gprint(s)
    print("done")



