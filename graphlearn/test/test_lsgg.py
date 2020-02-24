#!/usr/bin/env python

"""Provides the graph grammar class."""

#from graphlearn import lsgg_core_interface_pair as lcip
#from graphlearn import local_substitution_graph_grammar as lsgg


import logging
from graphlearn.util import util
import networkx as nx

import sys
logging.basicConfig(stream=sys.stdout, level=5) 
logger = logging.getLogger(__name__)


'''
def test_fit():
    lsggg = util.test_get_grammar()
    assert (4 == sum(len(e) for e in lsggg.productions.values()))
    # gprint( [e.graph for e in lsggg.productions[49532].values() ])
    # gprint( [e.graph for e in lsggg.productions[29902].values() ])


def test_extract_cip():
    graph = nx.path_graph(4)
    util._edenize_for_testing(graph)
    res = lsgg_core_interface_pair.make_cip(root_node=3, graph=graph, radius=1, thickness=1)
    # gprint(res.graph)
    assert ('cor' in str(res))


def test_neighbors():
    # make a grammar
    lsgg = util.test_get_grammar()

    # make agraph
    g = nx.path_graph(4)
    g = util._edenize_for_testing(g)
    g.nodes[3]['label'] = '5'
    stuff = list(lsgg.neighbors(g))
    assert (6 == len(stuff))
'''

def test_some_neighbors():
    # make a grammar
    lsgg = util.test_get_grammar()
    # make a graph
    g = nx.path_graph(4)
    g = util._edenize_for_testing(g)

    import structout as so
    so.gprint(lsgg.neighbors(g).__next__())


    #g.nodes[3]['label'] = '5'
    #assert (1 == len(list(lsgg.neighbors_sample(g, 1))))
    #assert (2 == len(list(lsgg.neighbors_sample(g, 2))))
    #assert (3 == len(list(lsgg.neighbors_sample(g, 3))))
    # gprint(list( lsgg.some_neighbors(g,1) ))
    # gprint(list( lsgg.some_neighbors(g,2) ))
    # gprint(list( lsgg.some_neighbors(g,3) ))


def test_pathgraphs_thin_interface():
    from graphlearn import LSGG 
    lsgg = LSGG(radii=[2], thickness = 1, nodelevel_radius_and_thickness = False)
    g = nx.path_graph(5)
    g.add_edge(2,4)
    g = util._edenize_for_testing(g)

    lsgg.fit([g,g,g])
    # there are 2 interfaces and 4 cores
    #assert lsgg.size()[0] == 2 
    #assert lsgg.size()[1] == 4 

    return lsgg,g 


def visualize_test_pathgraphs_thin_interface():
    lsgg,g = test_pathgraphs_thin_interface()
    print("GRAMMAR:")
    lsgg.structout()
    import structout as so 
    print("APPLYING ALL PRODUCTIONS:")
    so.gprint(list(lsgg.neighbors(g)))

